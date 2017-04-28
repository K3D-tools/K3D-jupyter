from ._version import version_info, __version__

from .k3d import *

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'k3d',
        'require': 'k3d/extension'
    }]
