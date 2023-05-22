# To release a new version of K3D on PyPI:

git add and git commit
rm -rf build
rm -rf dist
rm -rf k3d/static
rm -rf k3d/labextension
rm -rf js/dist
python -m build .
twine upload dist/*
cd js
grunt build
npm publish
cd ../docs
make html
