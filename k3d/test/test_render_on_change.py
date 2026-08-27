"""auto_rendering was renamed to render_on_change in 3.0.0.

The old name read like a switch for a render loop. There has never been one: K3D draws when
something changed and nothing else, so what the flag really gates is whether adding or updating
an object draws a frame by itself. The old name keeps working, loudly.
"""
import pytest

import k3d


def test_the_new_name_is_the_synced_trait():
    plot = k3d.plot()

    assert 'render_on_change' in plot.trait_names()
    assert 'auto_rendering' not in plot.trait_names()
    assert plot.render_on_change is True

    plot.render_on_change = False

    assert plot.render_on_change is False


def test_the_factory_takes_the_new_name():
    assert k3d.plot(render_on_change=False).render_on_change is False


def test_the_old_name_still_reads_and_writes_through_a_warning():
    plot = k3d.plot()

    with pytest.warns(DeprecationWarning, match='render_on_change'):
        plot.auto_rendering = False

    assert plot.render_on_change is False

    with pytest.warns(DeprecationWarning, match='render_on_change'):
        assert plot.auto_rendering is False


def test_the_old_name_still_works_as_a_factory_argument():
    with pytest.warns(DeprecationWarning, match='render_on_change'):
        plot = k3d.plot(auto_rendering=False)

    assert plot.render_on_change is False


def test_the_old_name_still_works_as_a_constructor_argument():
    with pytest.warns(DeprecationWarning, match='render_on_change'):
        plot = k3d.plot.__globals__['Plot'](auto_rendering=False)

    assert plot.render_on_change is False
