import os
import re
import codecs
from setuptools import setup, find_packages

# Single-source the version from bicky/__init__.py
here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')
#

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='bicky',
    version=find_version('bicky', '__init__.py'),
    description='find BICs for 1D photonic crystals slab',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Momchil Minkov',
    author_email='541361425@qq.com',
    url='https://github.com/fancompute/legume',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
