pip install -ve .
jupyter nbextension install --py --user k3d
jupyter nbextension enable --py --user k3d
@REM python setup.py sdist
@REM python setup.py bdist_wheel