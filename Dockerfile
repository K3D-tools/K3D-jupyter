FROM python:3.9.16-slim

SHELL ["/bin/bash", "--login", "-c"]

# update and install have to share a layer: a cached `apt-get update` goes stale as soon as the
# distro moves package versions, and the install then 404s on files the cached index still lists.
RUN apt-get update && apt-get install -y -qq \
        curl wget unzip libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Chrome and chromedriver are pinned together.
#
# The image-comparison references in k3d/test/references are tied to a specific rasterizer, so a
# Chrome that moves with each rebuild silently invalidates them - text and line antialiasing
# shift. Google's apt repo and the _current_ .deb only ever carry the newest release, so the
# pinned build comes from Chrome for Testing, which is an archive.
#
# Raising this version is expected to require regenerating the reference images.
ARG CHROME_VERSION=144.0.7559.109
ARG CFT=https://storage.googleapis.com/chrome-for-testing-public

# One layer, with its own apt-get update: the stable .deb is installed purely so that apt
# resolves Chrome's runtime libraries for us (deriving them from the package beats maintaining
# the list by hand), and the pinned Chrome for Testing binaries then replace what it provided.
RUN apt-get update \
    && wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && (dpkg --install google-chrome-stable_current_amd64.deb || apt-get -f install -y) \
    && rm google-chrome-stable_current_amd64.deb \
    && wget -q -O /tmp/chrome.zip "${CFT}/${CHROME_VERSION}/linux64/chrome-linux64.zip" \
    && unzip -q /tmp/chrome.zip -d /opt && rm /tmp/chrome.zip \
    && ln -sf /opt/chrome-linux64/chrome /usr/bin/google-chrome \
    && wget -q -O /tmp/chromedriver.zip "${CFT}/${CHROME_VERSION}/linux64/chromedriver-linux64.zip" \
    && unzip -q /tmp/chromedriver.zip -d /opt && rm /tmp/chromedriver.zip \
    && ln -sf /opt/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && rm -rf /var/lib/apt/lists/* \
    && google-chrome --version && chromedriver --version


RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
RUN nvm install v22
RUN npm install -g webpack webpack-cli

COPY requirements.txt .
RUN pip install -r requirements.txt
# chromedriver comes from Chrome for Testing above, pinned to the same build as Chrome, so
# chromedriver-binary is deliberately absent: nothing imported it, and the wheel it installed
# tracked a different Chrome major than the one in this image.
# jupyterlab and hatch-jupyter-builder are what `jupyter labextension build` and the wheel build
# run on; without them `npm run build:prod` and `python -m build` both fail in this image.
RUN pip install pytest pixelmatch flask selenium webdriver-manager scikit-image vtk build twine \
        jupyterlab hatch-jupyter-builder

# The source tree arrives as a bind mount, so this link dangles at build time and resolves once
# the mount is in place - JupyterLab reads it when the server starts. Doing it here rather than
# with `jupyter labextension develop` keeps it across containers: `docker compose run --rm` gives
# a fresh filesystem every time, and without the link every session reports
# "No version of module k3d is registered".
RUN mkdir -p /usr/local/share/jupyter/labextensions \
    && ln -s /opt/app/src/k3d/labextension /usr/local/share/jupyter/labextensions/k3d

WORKDIR /opt/app/src

CMD ["/bin/bash"]
