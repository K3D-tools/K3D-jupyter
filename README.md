# K3D Jupyter

Jupyter notebook extension for K3D visualization library.

## Try it Now!
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/K3D-tools/K3D-jupyter)

## Requirements

* [bower](http://bower.io/#install-bower) (to fetch K3D dependency)
* [pip](https://pypi.python.org/pypi/pip) (to install Python module)

## Installation

### Locally (as user)

```console
make install-user
```

### Locally (global)

```console
make install-global
```

> Both `install-user` and `install-global` targets support optional PIP parameter in order to provide pip executable to use (e.g. `PIP=pip3` may be used for Python3)

### Using vagrant

```console
vagrant up
make install-vagrant
```
