import glob
import os

from setuptools import setup, find_packages

data_files = [os.path.relpath(file, 'pongrid') for file in glob.glob('pongrid/data/**', recursive=True)]


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pongrid',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/sarashenyy/PonGrid',
    license='MIT',
    author='Yueyue Shen',
    author_email='shenyy@nao.cas.cn',
    description='simple tool for computing probability on parameter grid and visualization',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'pongrid': 'pongrid'},
    # include_package_data=True,
    # package_data={'': ['LICENSE', 'README.md'],
    #               'starcat': ['data/*']},
    package_data={'': ['LICENSE', 'README.md'],
                  'pongrid': data_files},
    install_requires=requirements
)
