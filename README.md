# K3D Jupyter

[![Downloads](https://pepy.tech/badge/k3d)](https://pepy.tech/project/k3d)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/k3d/badges/downloads.svg)](https://anaconda.org/conda-forge/k3d)
[![Build Status](https://travis-ci.org/K3D-tools/K3D-jupyter.svg)](https://travis-ci.org/K3D-tools/K3D-jupyter)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/alerts/)
[![Language Grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:javascript)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:python)

Jupyter notebook extension for 3D visualization.

![points_cloud](imgs/points_cloud.gif)

![streamlines](imgs/streamlines.gif)

![volume_rendering](imgs/vr.gif)

#### YouTube:

[![Volume renderer](https://img.youtube.com/vi/zCeQ_ZXy_Ps/0.jpg)](https://www.youtube.com/watch?v=zCeQ_ZXy_Ps)

[![Volume renderer](https://img.youtube.com/vi/9evYSq3ieVs/0.jpg)](https://www.youtube.com/watch?v=9evYSq3ieVs)

[![Volume renderer](https://img.youtube.com/vi/DbCiauTuJrU/0.jpg)](https://www.youtube.com/watch?v=DbCiauTuJrU)

[![Volume renderer](https://img.youtube.com/vi/wIbBpUlB5vc/0.jpg)](https://www.youtube.com/watch?v=wIbBpUlB5vc)


## Try it Now!

Interactive showcase gallery: [![Documentation Status](https://readthedocs.org/projects/k3d-jupyter/badge/?version=latest)](https://k3d-jupyter.readthedocs.io/en/latest/showcase/index.html)

Documentation is generated at readthedocs: [![Documentation Status](https://readthedocs.org/projects/k3d-jupyter/badge/?version=latest)](https://k3d-jupyter.readthedocs.io/en/latest/?badge=latest)

Jupyter version: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/K3D-tools/K3D-jupyter/master?filepath=index.ipynb)

## Installation

### PyPI

To install from PyPI use pip:

    $ pip install k3d

### Conda/Anaconda

To install from conda-forge use:

    $ conda install -c conda-forge k3d

### Installing directly from GitHub

To install directy from this repository (requires git and node.js + npm to build):

    $ pip install git+https://github.com/K3D-tools/K3D-jupyter

This also makes possible installing the most up-to-date development version (same requirements):

    $ pip install git+https://github.com/K3D-tools/K3D-jupyter@devel

To install any historical version, replace `devel` above with any tag or commit hash.

### Source

For a development installation (requires npm and node.js),

    $ git clone https://github.com/K3D-tools/K3D-jupyter.git
    $ cd K3D-jupyter
    $ pip install -e .

Then, if required, JupyterLab installation:

    $ jupyter labextension install ./js

## JupyterLab

Then, if required, JupyterLab installation:

    $ jupyter labextension install k3d

Please notice that support for jupyterLab is still experimental.

### Developer's How To

Please make sure to take a look at the [HOW-TO.md](HOW-TO.md) document.
