"""Plugin registry and the built-in reference method."""

import os
import sys
import types
from importlib.metadata import EntryPoint

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memseg import segmenters  # noqa: E402
from memseg.cli import main  # noqa: E402
from memseg.domain import MembraneMask  # noqa: E402
from memseg.segmenters.base import SegmenterPlugin  # noqa: E402
from memseg.segmenters.components import ConnectedComponentsSegmenter  # noqa: E402


def test_components_one_instance_per_piece_and_drops_specks():
    m = np.zeros((40, 40), bool)
    m[5:10, 5:30] = True       # 125 px
    m[20:30, 20:22] = True     # 20 px
    m[35, 35] = True           # 1 px speck
    res = ConnectedComponentsSegmenter(min_size=10).segment(MembraneMask(m))
    assert res.labels.n_instances == 2
    assert res.info["components"] == 3
    assert set(np.unique(res.labels.data)) == {0, 1, 2}


class _FakePlugin(SegmenterPlugin):
    name = "fake"
    summary = "fake method from an entry point"

    def build(self, args, overrides):
        return ConnectedComponentsSegmenter()


def test_entry_point_methods_are_discovered(monkeypatch, capsys):
    mod = types.ModuleType("fake_memseg_plugin")
    mod.FakePlugin = _FakePlugin
    monkeypatch.setitem(sys.modules, "fake_memseg_plugin", mod)
    eps = [EntryPoint(name="fake", value="fake_memseg_plugin:FakePlugin", group=segmenters.ENTRY_POINT_GROUP)]
    monkeypatch.setattr(segmenters, "entry_points", lambda group: eps if group == segmenters.ENTRY_POINT_GROUP else [])

    assert segmenters.available() == ["components", "fake"]
    assert isinstance(segmenters.get_plugin("fake"), _FakePlugin)
    assert main(["methods"]) == 0
    assert "fake method from an entry point" in capsys.readouterr().out


def test_builtin_name_wins_over_entry_point(monkeypatch):
    eps = [EntryPoint(name="components", value="nowhere:Nothing", group=segmenters.ENTRY_POINT_GROUP)]
    monkeypatch.setattr(segmenters, "entry_points", lambda group: eps)
    assert segmenters.get_plugin("components").name == "components"


def test_broken_plugin_does_not_break_methods_listing(monkeypatch, capsys):
    eps = [EntryPoint(name="broken", value="no_such_module_xyz:Plugin", group=segmenters.ENTRY_POINT_GROUP)]
    monkeypatch.setattr(segmenters, "entry_points", lambda group: eps)
    assert main(["methods"]) == 0
    assert "failed to load" in capsys.readouterr().out
