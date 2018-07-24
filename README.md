# K3D Jupyter

[![Build Status](https://travis-ci.org/K3D-tools/K3D-jupyter.svg)](https://travis-ci.org/K3D-tools/K3D-jupyter)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/alerts/)
[![Language Grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:javascript)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:python)

Jupyter notebook extension for 3D visualization.

![screenshot](screenshot.png)

## Try it Now!
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/K3D-tools/K3D-jupyter/master?filepath=index.ipynb)

## Installation

To install use pip:

    $ pip install k3d
    $ jupyter nbextension install --py --sys-prefix k3d
    $ jupyter nbextension enable --py --sys-prefix k3d
    $ jupyter labextension install k3d

For a development installation (requires npm and node.js),

    $ git clone https://github.com/K3D-tools/K3D-jupyter.git
    $ cd K3D-jupyter
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --sys-prefix k3d
    $ jupyter nbextension enable --py --sys-prefix k3d
    $ jupyter labextension install ./js

Please note that the `ipywidgets` extension needs to be
[installed and enabled](http://ipywidgets.readthedocs.io/en/latest/user_install.html).

## How to

Please make sure to take a look at the [HOW-TO.md](HOW-TO.md) document.
