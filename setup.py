#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
from jupyterpip import cmdclass
import json


def get_version(bower_path):
    with open(bower_path) as bower:
        return json.load(bower)['version']

install = cmdclass('k3d_widget')['install']
version = get_version('bower.json')
k3d_version = get_version('k3d_widget/lib/k3d/bower.json')


class InstallK3D(install):  # pylint: disable=no-init
    def run(self):
        print('Creating version file...')
        with open('k3d/version.py', 'w') as version_file:
            version_file.write('version = %s\n' % json.dumps({'k3d-lib': k3d_version, 'k3d-jupyter': version}))
        install.run(self)

setup(
    name='k3d',
    version=version,
    packages=['k3d'],
    include_package_data=True,
    install_requires=['jupyter-pip'],
    cmdclass={
        'install': InstallK3D
    },
)
