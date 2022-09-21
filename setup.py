"""
k3d setup
"""
import json
from pathlib import Path

from jupyter_packaging import (
    create_cmdclass,
    install_npm,
    ensure_targets,
    combine_commands,
    skip_if_exists
)
import setuptools

HERE = Path(__file__).parent.resolve()

# The name of the project
name = "k3d"

lab_path = (HERE / name / "labextension")

# Representative files that should exist after a successful build
jstargets = [
    str(lab_path / "package.json"),
]

package_data_spec = {
    name: ["*"],
}

node_root = HERE / 'js'

labext_name = "k3d"

data_files_spec = [
    ("share/jupyter/labextensions/%s" % labext_name, str(lab_path), "**"),
    ("share/jupyter/labextensions/%s" % labext_name, str(HERE), "install.json"),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'standalone.js'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'snapshot_standalone.txt'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'snapshot_online.txt'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'snapshot_inline.txt'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'headless.html'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'extension.js'),
    ("k3d/static/%s" % labext_name, str(HERE / 'k3d' / 'static'), 'index.js')
]

cmdclass = create_cmdclass("jsdeps",
    package_data_spec=package_data_spec,
    data_files_spec=data_files_spec
)

js_command = combine_commands(
    install_npm(path=node_root, build_dir=node_root, build_cmd="build"),
    install_npm(HERE, build_cmd="build:prod", npm=["jlpm"]),
    ensure_targets(jstargets),
)

is_repo = (HERE / ".git").exists()
if is_repo:
    cmdclass["jsdeps"] = js_command
else:
    cmdclass["jsdeps"] = skip_if_exists(jstargets, js_command)

long_description = (HERE / "README.md").read_text()

# Get the package info from package.json
pkg_json = json.loads((HERE / "package.json").read_bytes())

setup_args = dict(
    name=name,
    version=pkg_json["version"],
    url=pkg_json["homepage"],
    author=pkg_json["author"]["name"],
    author_email=pkg_json["author"]["email"],
    description=pkg_json["description"],
    license=pkg_json["license"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass=cmdclass,
    packages=setuptools.find_packages(),
    install_requires=[
        "traittypes",
        "ipywidgets>=7,<9",
        "traitlets",
        "numpy",
        "msgpack"
    ],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.6",
    platforms="Linux, Mac OS X, Windows",
    keywords=["Jupyter", "JupyterLab", "JupyterLab3"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
    ],
)

if __name__ == "__main__":
    setuptools.setup(**setup_args)
