# To release a new version of K3D on PyPI:

docker compose run --rm --service-ports k3d-build bash

git add and git commit
rm -rf build
rm -rf dist
rm -rf k3d/static
rm -rf js/dist
python -m build .
# the hatch-jupyter-builder hook reruns the webpack build, so k3d/static and
# js/dist are regenerated; build the wheel BEFORE anything that needs a fresh
# editable install, or rebuild js afterwards (cd js && npm run build)
twine upload dist/*
cd js
npm publish
# prepublishOnly runs the webpack build and the manifest check
cd ../docs
make html
