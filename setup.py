# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    _license = f.read()

setup(
    name='sequence_modeling',
    version='0.1.0',
    description='Deep leaning model to classify short sentence.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='ushioasahi@keio.jp',
    license=_license,
    packages=find_packages(exclude=('tests', 'data', 'LSTM_Encoder')),
    # package_data={'': ['*.json']},
    include_package_data=True,
    install_requires=[],
    test_suite='tests'
)
