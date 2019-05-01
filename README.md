# K3D Jupyter

[![Build Status](https://travis-ci.org/K3D-tools/K3D-jupyter.svg)](https://travis-ci.org/K3D-tools/K3D-jupyter)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/alerts/)
[![Language Grade: JavaScript](https://img.shields.io/lgtm/grade/javascript/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:javascript)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/K3D-tools/K3D-jupyter.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/K3D-tools/K3D-jupyter/context:python)

Jupyter notebook extension for 3D visualization.

![points_cloud](imgs/points_cloud.gif)

![streamlines](imgs/streamlines.gif)

![volume_rendering](imgs/vr.gif)

#### YouTube:

[![Volume renderer](https://img.youtube.com/vi/4XMyZWviSsU/0.jpg)](https://www.youtube.com/watch?v=4XMyZWviSsU)

[![Volume renderer](https://img.youtube.com/vi/Hcr8lf-fawU/0.jpg)](https://www.youtube.com/watch?v=Hcr8lf-fawU)

[![Volume renderer](https://img.youtube.com/vi/ZFkw8zt2QXs/0.jpg)](https://www.youtube.com/watch?v=ZFkw8zt2QXs)


## Try it Now!
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/K3D-tools/K3D-jupyter/master?filepath=index.ipynb)

## Installation

### PyPI

To install from PyPI use pip:

    $ pip install k3d

Then, if required, JupyterLab installation:

    $ jupyter labextension install k3d

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

### Developer's How To

Please make sure to take a look at the [HOW-TO.md](HOW-TO.md) document.
