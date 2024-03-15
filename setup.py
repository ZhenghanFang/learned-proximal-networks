import os
from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="lpn",
    packages=find_packages(include=["lpn", "lpn.*"]),
    version="0.1.0",
    description="",
    author="Zhenghan Fang",
    long_description=read("README.md"),
)
