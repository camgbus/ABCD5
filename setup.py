from setuptools import setup, find_packages

setup(
    name='ABCD5',
    version='0.1',
    description='Working with the ABCD 5.0 release.',
    url='https://github.com/camgbus/ABCD5',
    keywords='python setuptools',
    packages=find_packages(include=['abcd', 'abcd.*']),
)