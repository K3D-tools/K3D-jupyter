import pytest

import k3d
from .plot_compare import prepare, compare


def test_text2d():
    prepare()

    pytest.plot += k3d.text2d(
        '\\int_{-\\infty}^\\infty \\hat f(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi',
        [0.75, 0.5], color=0, size=1.5, reference_point='lt')

    pytest.plot += k3d.text2d('{(1,1,\\frac{5}{\\pi})}', [0.25, 0.5],
                              color=0, size=1.5, reference_point='rb')
    pytest.plot += k3d.text2d('<h1 style="color: red;">Important!</h1>Hello<strong>World</strong>',
                              [0.5, 0.5],
                              color=0, size=1.5, is_html=True, reference_point='rb')

    compare('text2d', False)


def test_text2d_without_box():
    prepare()

    pytest.plot += k3d.text2d(
        '\\int_{-\\infty}^\\infty \\hat f(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi',
        [0.75, 0.5], color=0, size=1.5, reference_point='lt', label_box=False)

    pytest.plot += k3d.text2d('{(1,1,\\frac{5}{\\pi})}', [0.25, 0.5],
                              color=0, size=1.5, reference_point='rb', label_box=False)
    pytest.plot += k3d.text2d('<h1 style="color: red;">Important!</h1>Hello<strong>World</strong>',
                              [0.5, 0.5],
                              color=0, size=1.5, is_html=True, reference_point='rb',
                              label_box=False)

    compare('text2d_without_box', False)
