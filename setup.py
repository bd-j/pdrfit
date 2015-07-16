#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="pdrfit",
    version='0.1.0',
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["pdrfit"],
    url="https://github.com/bd-j/pdrfit",
    license="LICENSE",
    description="Tools for fitting PDR models",
    long_description=open("README.md").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n",
                   # + open("HISTORY.rst").read(),
    package_data={"pdrfit": ["data/*txt"]},
    include_package_data=True,
    install_requires=["numpy", "scipy >= 0.9", "astropy", "matplotlib"],
)
