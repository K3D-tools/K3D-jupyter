#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

try:
    from jupyterpip import cmdclass
except:
    import pip, importlib
    pip.main(['install', 'jupyter-pip']); cmdclass = importlib.import_module('jupyterpip').cmdclass

setup(
    name='k3d',
    version='0.2.1',
    packages=['k3d'],
    include_package_data=True,
    install_requires=['jupyter-pip'],
    cmdclass=cmdclass('k3d_widget'),
)
