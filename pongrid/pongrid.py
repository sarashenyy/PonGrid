import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

module_dir = os.path.dirname(__file__)
style_file = os.path.join(module_dir, 'data/mystyle.mplstyle')

plt.style.use(style_file)
cmap = ListedColormap(sns.color_palette("RdBu_r", n_colors=256))


class PonGrid(object):
    def __init__(self,
                 param_num,
                 param_name,
                 param_range
                 ):
        """

        Parameters
        ----------
        param_num : int
        param_name : list
        """
        self.param_num = param_num
        self.param_name = param_name
        self.param_range = param_range
        self.param_grid, self.len_in_each = self.get_param_grid()
        self.args_list = self.get_args_list()

        self.joint_log_posterior = None
        self.joint_posterior = None

    def get_param_grid(self):
        """

        Parameters
        ----------
        self.param_range : list, shape=(param_num, 3)
            [[start, end, step], [start, end, step], ...]

        Returns
        -------
        param_grid : list
        """
        param_grid = []
        length = []
        total_samples = 1
        for i in range(self.param_num):
            aux_start, aux_end, aux_step = self.param_range[i][0], self.param_range[i][1], self.param_range[i][2]
            aux_grid = np.arange(start=aux_start, stop=aux_end, step=aux_step)
            aux_len = len(aux_grid)
            param_grid.append(aux_grid)
            length.append(aux_len)
            print(f'{self.param_name[i]} len: {aux_len}, grid: {aux_grid}')
            total_samples *= aux_len

        print(f'total samples: {total_samples}')
        return param_grid, length

    def get_args_list(self):
        """

        Returns
        -------
        args_list

        """
        id_list = np.indices(self.len_in_each)
        param_vals = []
        for i in range(self.param_num):
            aux_grid = self.param_grid[i]
            aux_id = id_list[i]
            aux_id = aux_id.ravel()

            aux_vals = aux_grid[aux_id]
            param_vals.append(aux_vals)

        args_list = []
        for i in range(len(param_vals[0])):
            aux_arg = []
            for j in range(self.param_num):
                aux_val = param_vals[j][i]
                aux_arg.append(aux_val)
            args_list.append(aux_arg)

        return args_list

    def run_grid(self,
                 log_posterior,
                 processes=None):
        # joint_log_posterior = np.zeros(self.len_in_each)
        if processes is None:
            processes = cpu_count()
        with Pool(processes) as pool:
            with tqdm(total=len(self.args_list)) as progress_bar:
                results = []
                for result in pool.imap(log_posterior, self.args_list):
                    results.append(result)
                    progress_bar.update(1)

        results = np.array(results)
        joint_log_posterior = results.reshape(self.len_in_each)
        self.joint_log_posterior = joint_log_posterior
        return joint_log_posterior

    def check_log_posterior(self,
                            shift=True,
                            shifted_to=10):
        """
        check the log(posterior) value range and calculate posterior.
        Because np.exp() can only maintain calculation accuracy in [-745, 705],
        we need to shift the log(posterior) value by adding a max number to the whole
        if they are smaller than -745.
        The recommand shift value is 10.

        Parameters
        ----------
        shift : bool
        shifted_to : int
            shift the whole log(posterior) to maximum(log(posterior)) = shifted_to, default is 10.

        Returns
        -------

        """
        joint_loglike_noinf = self.joint_log_posterior[self.joint_log_posterior != -np.inf]
        print(f'joint log(posterior) without inf: [{np.min(joint_loglike_noinf)}, {np.max(joint_loglike_noinf)}]')
        max_loglike = np.max(self.joint_log_posterior)
        if shift:
            print(f'shifted log(posterior) to {shifted_to}')
            joint_loglike_shifted = (self.joint_log_posterior - max_loglike) + shifted_to
            print(f'shifted joint log(posterior): [{np.min(joint_loglike_shifted[joint_loglike_shifted != -np.inf])}, '
                  f'{np.max(joint_loglike_shifted)}]')

            joint_posterior = np.exp(joint_loglike_shifted)
            print(f'joint posterior: [{np.min(joint_posterior)}, {np.max(joint_posterior)}]')

            self.joint_posterior = joint_posterior
            return joint_posterior, joint_loglike_shifted

        else:
            joint_posterior = np.exp(self.joint_log_posterior)
            print(f'joint posterior: [{np.min(joint_posterior)}, {np.max(joint_posterior)}]')

            self.joint_posterior = joint_posterior
            return joint_posterior

    def show_grid_probability(self,
                              figpath,
                              labels,
                              truths=None,
                              joint_posterior=None):
        if self.joint_posterior is None and joint_posterior is None:
            print('Please run pg.check_log_posterior() first to '
                  'ensure that the posterior value satisfies the calculation accuracy of np.exp()!!')
        # if you want to draw the already exist posteriror grid, set your joint_posterior here
        if joint_posterior is not None:
            self.joint_posterior = joint_posterior

        num_subplot = len(self.joint_posterior.shape)

        fig = plt.figure(figsize=(15, 17))  # (Width, height)
        gs = gridspec.GridSpec(num_subplot, num_subplot)  # (nrows,ncols)

        rows_to_draw = np.arange(num_subplot)
        for col in range(num_subplot):

            # ! calculate marginal distribution of single parameter
            # get x axis data
            data_x = self.param_grid[col]
            # get y axis data (marginal distribution of p)
            marginal_p = np.sum(self.joint_posterior, axis=tuple(set(np.arange(num_subplot)) - {col}))
            # normalize
            marginal_p = marginal_p / np.sum(marginal_p)
            # ? draw diagonal plots with marginal distribution of single parameter
            ax_diagonal = plt.subplot(gs[col, col])
            ax_diagonal.plot(data_x, marginal_p, c='k')
            ax_diagonal.scatter(data_x, marginal_p, c='grey')
            if truths is not None:
                ax_diagonal.axvline(truths[col], c='r', linestyle='--')

            # put y ticks to right
            ax_diagonal.yaxis.tick_right()
            ax_diagonal.xaxis.set_label_position('top')
            ax_diagonal.set_xlabel(labels[col], fontsize=16)
            # to control the x extent in non-diagonal subplots
            left, right = ax_diagonal.get_xlim()

            # ! calculate marginal distribution of two parameters
            if col < num_subplot - 1:
                rows_to_draw = rows_to_draw[1:]
                for row in rows_to_draw:
                    marginal_pp = np.sum(self.joint_posterior, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
                    ln_marginal_pp = np.log(marginal_pp)
                    ln_marginal_pp = np.transpose(ln_marginal_pp)
                    # print(f'row,col={row, col}, axis={tuple(set(np.arange(num_subplot)) - {col, row})}, '
                    #       f'marginal_pp={marginal_pp.shape}')
                    data_y = self.param_grid[row]
                    x_grid, y_grid = np.meshgrid(data_x, data_y)

                    # ? draw non diagonal plots with marginal distribution of two parameters
                    ax = plt.subplot(gs[row, col])
                    ax.scatter(x_grid, y_grid, color='gray', alpha=0.5, s=2)
                    # to control the y extent in non-diagonal subplots
                    bottom, top = ax.get_ylim()

                    # * NOTE! ax.contour(X,Y,Z)
                    # * NOTE! X and Y must both be 2D with the same shape as Z (e.g. created via numpy.meshgrid)
                    ax.contour(x_grid, y_grid, ln_marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                               extent=(left, right, bottom, top),
                               origin='lower')

                    # * NOTE! ax.imshow(pp), pp.shape=(rows_Y, cols_X)
                    # * NOTE! ax.imshow(origin='lower')
                    # *       control which point in pp represents the original point in figure(lower / upper)
                    im = ax.imshow(ln_marginal_pp, cmap=cmap,
                                   extent=(left, right, bottom, top),
                                   origin='lower')
                    cbar = plt.colorbar(im, ax=ax, location='top')
                    cbar.ax.tick_params(labelsize=8)
                    cbar.formatter.set_useOffset(False)
                    cbar.update_ticks()

                    ax.set_aspect('auto')
                    if truths is not None:
                        ax.axvline(truths[col], c='r', linestyle='--')
                        ax.axhline(truths[row], c='r', linestyle='--')
                        ax.plot(truths[col], truths[row], 'sr')

                    if row != num_subplot - 1:
                        ax.set_xticklabels([])
                    if col != 0:
                        ax.set_yticklabels([])

                    if row == num_subplot - 1:
                        ax.set_xlabel(labels[col], fontsize=16)
                        ax.tick_params(axis='x', labelsize=8)
                    if col == 0:
                        ax.set_ylabel(labels[row], fontsize=16)
                        ax.tick_params(axis='y', labelsize=8)

        plt.savefig(figpath, bbox_inches='tight')
        plt.close(fig)
