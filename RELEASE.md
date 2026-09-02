# To release a new version of K3D on PyPI:

# Before anything else - both linters and the whole suite, in the container. CI runs the
# same two linters, so this is the local shortcut rather than the only gate.
# The suite has to run in docker: the visual references are tied to the Chrome pinned in
# the image, and a host Chrome differs by enough pixels to fail every visual test.
docker compose run --rm k3d-build bash -lc "cd /opt/app/src && python -m ruff check ."
docker compose run --rm k3d-build bash -lc "cd /opt/app/src/js && npx grunt codeStyle"
docker compose run --rm k3d-build bash -lc "cd /opt/app/src/k3d && python -m pytest"

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

# Then the GitHub Release, from main, after the devel -> main PR is merged. This is what
# archives the release: the Zenodo webhook listens for `release` events only, and never
# backfills, so a version released without this step is uncitable.
gh release create vX.Y.Z --repo K3D-tools/K3D-jupyter --target main --title "vX.Y.Z" --generate-notes

# Confirm Zenodo picked it up (a few minutes):
curl -s https://zenodo.org/api/records/3247652 | python -c "import json,sys; m=json.load(sys.stdin)['metadata']; print(m.get('version'), m.get('publication_date'))"

# Finally bump `version` and `date-released` in CITATION.cff to the version just published.
