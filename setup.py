from __future__ import print_function

import os

from setuptools import setup, find_packages

from distutils import log

from setupbase import (
    create_cmdclass, install_npm, combine_commands, BaseCommand,
    ensure_targets
)

here = os.path.dirname(os.path.abspath(__file__))
node_root = os.path.join(here, 'js')

log.set_verbosity(log.DEBUG)
log.info('setup.py entered')
log.info('$PATH=%s' % os.environ['PATH'])

LONG_DESCRIPTION = 'Jupyter notebook extension for 3D visualization.'

# Representative files that should exist after a successful build
targets = [
    os.path.join(here, 'k3d', 'static', 'extension.js'),
    os.path.join(here, 'k3d', 'static', 'index.js')
]

cmdclass = create_cmdclass(('jsdeps',))
cmdclass['jsdeps'] = combine_commands(
    install_npm(node_root),
    ensure_targets(targets)
)

version_ns = {}
with open(os.path.join(here, 'k3d', '_version.py')) as f:
    exec(f.read(), {}, version_ns)

setup_args = {
    'name': 'K3D',
    'version': version_ns['__version__'],
    'license': 'MIT',
    'description': 'Jupyter notebook extension for 3D visualization.',
    'long_description': LONG_DESCRIPTION,
    'include_package_data': True,
    'data_files': [
        ('share/jupyter/nbextensions/k3d', [
            'k3d/static/*'
        ]),
    ],
    'install_requires': [
        'ipywidgets>=7.0.1',
        'traittypes',
        'traitlets',
        'numpy>=1.11.0'
    ],
    'packages': find_packages(),
    'zip_safe': False,
    'cmdclass': cmdclass,
    'author': 'k3d team',
    'author_email': 'artur.trzesiok@gmail.com',
    'url': 'http://jupyter.org',
    'keywords': [
        'ipython',
        'jupyter',
        'widgets',
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
}

setup(**setup_args)
