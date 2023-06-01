# K3D Jupyter

[![Downloads](https://pepy.tech/badge/k3d)](https://pepy.tech/project/k3d)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/k3d/badges/downloads.svg)](https://anaconda.org/conda-forge/k3d)
[![CodeQL](https://github.com/K3D-tools/K3D-jupyter/workflows/CodeQL/badge.svg)](https://github.com/K3D-tools/K3D-jupyter/actions)

<div>

<img src="https://k3d-jupyter.org/_static/logo.png" width="25%" align="right">

K3D lets you create 3D plots backed by WebGL with high-level API (surfaces, isosurfaces, voxels,
mesh, cloud points, vtk objects, volume renderer, colormaps, etc). The primary aim of K3D-jupyter is
to be easy for use as stand alone package like matplotlib, but also to allow interoperation with
existing libraries as VTK. K3D can be run as:

- Jupyter Notebook extension 🚀
- Jupyter Lab extension 🎉
- Google Colab extension 🧪 [still experimental]
- Standalone HTML/JS 📑

Documentation: [https://k3d-jupyter.org](https://k3d-jupyter.org)
</div>


## Showcase:

![points_cloud](imgs/points_cloud.gif)

![streamlines](imgs/streamlines.gif)

![volume_rendering](imgs/vr.gif)

![transfer_function_editor](imgs/tf_edit.gif)

### YouTube:

Click to watch at YouTube:

[![Volume renderer](https://i3.ytimg.com/vi/zCeQ_ZXy_Ps/maxresdefault.jpg)](https://www.youtube.com/watch?v=zCeQ_ZXy_Ps)

[![Volume renderer](https://i3.ytimg.com/vi/9evYSq3ieVs/maxresdefault.jpg)](https://www.youtube.com/watch?v=9evYSq3ieVs)

[![Volume renderer](https://i3.ytimg.com/vi/DbCiauTuJrU/maxresdefault.jpg)](https://www.youtube.com/watch?v=DbCiauTuJrU)

[![Volume renderer](https://i3.ytimg.com/vi/wIbBpUlB5vc/maxresdefault.jpg)](https://www.youtube.com/watch?v=wIbBpUlB5vc)

## Try it Now!

Watch: [Interactive showcase gallery](https://k3d-jupyter.org/gallery/index.html)

Jupyter
version: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/K3D-tools/K3D-jupyter/main?filepath=index.ipynb)

## Installation

### PyPI

To install from PyPI use pip:

    $ pip install k3d

### Conda/Anaconda

To install from conda-forge use:

    $ conda install -c conda-forge k3d

### Google Colab

First you need to install k3d:

    !pip install k3d
    !jupyter nbextension install --py --user k3d
    !jupyter nbextension enable --py --user k3d

After that you need to activate custom widgets and switch k3d to text protocol:

    import k3d
    from google.colab import output
    
    output.enable_custom_widget_manager()
    
    k3d.switch_to_text_protocol()

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

### Code of Conduct

K3D-jupyter follows the Python Software Foundation Code of Conduct in everything we do.

## Kudos

- Travis is ❤️
- OpenDreamKit is 🚀
- Three.js is 🥇

## Acknowledgments

<table class="none">
<tr>
<td>
<img src="http://opendreamkit.org/public/logos/Flag_of_Europe.svg" width="128">
</td>
<td>
Research Infrastructure project
This package was created as part of the Horizon 2020 European
<a href="https://opendreamkit.org/">OpenDreamKit</a>
(grant agreement <a href="https://opendreamkit.org/">#676541</a>).
</td>
</tr>
</table>

