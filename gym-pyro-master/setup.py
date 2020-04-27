import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.stderr.write('Python >= 3.7 is required.')
    sys.exit(1)


setup(
    name='gym_pyro',
    version='1.0.0',
    packages=find_packages(),
)
