"""
#
# scikit_optim setup.py
#
# Copyright(c) 2018, Antoine Emil Zambelli.
#
"""

from setuptools import setup, find_packages


version = '2.0.4'

setup(
    name='scikit_optim',
    version=version,
    url='https://github.com/antoinezambelli/scikit_optim',
    license='MIT',
    author='Antoine Zambelli',
    author_email='antoine.zambelli@gmail.com',
    description='Scikit CV',
    long_description='Scikit CV Convenience Library',
    packages=find_packages(
        exclude=(
           '.*',
           'EGG-INFO',
           '*.egg-info',
           '_trial*',
           "*.tests",
           "*.tests.*",
           "tests.*",
           "tests",
           "examples.*",
           "examples",
        )
    ),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)
