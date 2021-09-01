
conda create -y -n jupyterlab-ext --override-channels --strict-channel-priority -c conda-forge -c nodefaults jupyterlab=3 cookiecutter nodejs jupyter-packaging git numpy ipywidgets traittypes &&
conda activate jupyterlab-ext &&
pip install -ve . &&
jupyter nbextension install --py --user k3d &&
jupyter nbextension enable --py --user k3d &&
pip install build &&
cd dist &&
python -m build &&
cd - &&
jupyter lab

# 1. require i style -> refaktor do wersji bez idków
# 2. dostosowanie istniejącego repo - up