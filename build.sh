
conda create -y -n jupyterlab-ext --override-channels --strict-channel-priority -c conda-forge -c nodefaults jupyterlab=3 cookiecutter nodejs jupyter-packaging git numpy ipywidgets traittypes &&
conda activate jupyterlab-ext &&
#jupyter labextension install @jupyter-widgets/jupyterlab-manager
pip install -ve . &&
jupyter lab

# 1. require i style -> refaktor do wersji bez idków
# 2. dostosowanie istniejącego repo - 