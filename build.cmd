pip install -ve .
jupyter nbextension install --py --user k3d
jupyter nbextension enable --py --user k3d
python setup.py sdist
python setup.py bdist_wheel