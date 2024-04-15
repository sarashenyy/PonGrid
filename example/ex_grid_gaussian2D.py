import joblib
import numpy as np
from pongrid import PonGrid
from scipy.stats import multivariate_normal

# 生成二维正态分布数据
np.random.seed(42)
mu1, mu2 = 0, 1
sigma1, sigma2 = 2, 2
mu_true = [mu1, mu2]  # 真实均值
cov_true = [[sigma1 ** 2, 0], [0, sigma2 ** 2]]  # 真实协方差矩阵
data = np.random.multivariate_normal(mu_true, cov_true, size=5000)


# 定义 log likelihood 函数
def log_likelihood(theta, data):
    mu1, mu2, sigma1, sigma2 = theta
    cov = [[sigma1 ** 2, 0], [0, sigma2 ** 2]]
    return np.sum(multivariate_normal.logpdf(data, mean=[mu1, mu2], cov=cov))


# 定义 prior 函数
def log_prior(theta):
    mu1, mu2, sigma1, sigma2 = theta
    if sigma1 > 0 and sigma2 > 0:
        return 0.0
    return -np.inf


# 定义 log posterior 函数
def log_posterior_wrapin(theta, data):
    return log_prior(theta) + log_likelihood(theta, data)


def log_posterior(theta):
    return log_posterior_wrapin(theta, data)


if __name__ == '__main__':
    pg = PonGrid(
        param_num=4,
        param_name=['mu1', 'mu2', 'sigma1', 'sigma2'],
        param_range=[[-0.5, 0.5, 0.05],
                     [0.5, 1.5, 0.05],
                     [1.5, 2.5, 0.05],
                     [1.5, 2.5, 0.05]]
    )

    joint_log_posterior = pg.run_grid(
        log_posterior=log_posterior,
        processes=None) # if processes in None, use ALL your CPUs

    # save joint_log_likelihood if needed
    joblib.dump(joint_log_posterior, 'ex_gridpost_gaussian2D.joblib')
    # if you want to read
    # joint_log_likelihood = joblib.load('ex_gridpost_gaussian2D.joblib')

    joint_posterior, joint_posterior_shifted = pg.check_log_posterior(shift=True, shifted_to=10)

    pg.show_grid_probability(
        figpath='ex_gridres_gaussian2D.png',
        labels=['$\mu_1$', '$\mu_2$', '$\sigma_1$', '$\sigma_2$'],
        truths=[mu1, mu2, sigma1, sigma2]
    )

