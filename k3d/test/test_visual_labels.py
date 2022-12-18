import pytest

import k3d
from .plot_compare import prepare, compare


def test_labels():
    prepare()

    pytest.plot += k3d.label('Test dynamic', (0, 0, 0), mode='dynamic')
    pytest.plot += k3d.label('Test local', (1.5, 0, 0), mode='local')
    pytest.plot += k3d.label('Test side 1', (1.5, 1.5, 1.5), mode='side')
    pytest.plot += k3d.label('Test side 2', (0, 0, 1.5), mode='side', is_html=True)

    compare('labels', False)


def test_labels_without_box():
    prepare()

    pytest.plot += k3d.label('Test dynamic', (0, 0, 0), mode='dynamic', label_box=False)
    pytest.plot += k3d.label('Test local', (1.5, 0, 0), mode='local', label_box=False)
    pytest.plot += k3d.label('Test side 1', (1.5, 1.5, 1.5), mode='side', label_box=False)
    pytest.plot += k3d.label('Test side 2', (0, 0, 1.5), mode='side', is_html=True, label_box=False)

    compare('labels_without_box', False)
