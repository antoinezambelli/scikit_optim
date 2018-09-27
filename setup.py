"""
#
# scikit_optim setup.py
#
# Copyright(c) 2018, Carium, Inc. All rights reserved.
#
"""

from setuptools import setup


version = '1.0.0'

setup(
    name='scikit_optim',
    version=version,
    url='https://github.com/antoinezambelli/scikit_optim',
    license='MIT',
    author='Antoine Zambelli',
    author_email='antoine.zambelli@gmail.com',
    description='Scikit CV',
    long_description='Scikit CV Convenience Library',
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn'
    ]
)