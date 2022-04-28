.. _download:

download
========

.. autofunction:: k3d.helpers.download

.. note::
    If a file with a similar name already exists in the current directory,
    `download` will not overwrite the file and just return the file name.

**Examples**

.. code-block:: python3

    import k3d
    import numpy as np

    url = 'https://graylight-imaging.com/wp-content/themes/bootscore-child-main/img/logo/logo.svg'

    filename = k3d.helpers.download(url)

    """
    logo.svg
    """