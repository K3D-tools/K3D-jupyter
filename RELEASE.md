# To release a new version of K3D on PyPI:

Update _version.py (set release version, remove 'dev')
git add and git commit
python setup.py sdist upload
python setup.py bdist_wheel upload
cd js
grunt build
npm publish
cd ../docs
make html