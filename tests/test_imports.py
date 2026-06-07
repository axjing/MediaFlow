"""Smoke tests for the package surface area."""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import mediaflow


PUBLIC_SUBPACKAGES = ["mediaflow", "mediaflow.common", "mediaflow.tts", "mediaflow.clipper"]


@pytest.mark.parametrize("module_name", PUBLIC_SUBPACKAGES)
def test_subpackage_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_version_present() -> None:
    assert hasattr(mediaflow, "__version__")
    assert isinstance(mediaflow.__version__, str)


def test_cli_module_loads() -> None:
    cli = importlib.import_module("mediaflow.cli")
    assert callable(cli.main)


def test_main_module_loads() -> None:
    main = importlib.import_module("mediaflow.__main__")
    assert callable(main.main)


def test_common_surface() -> None:
    from mediaflow.common import (  # noqa: F401
        AttrDict,
        Logging,
        MD,
        YamlParser,
        add_cut,
        change_ext,
        check_exists,
        expand_segments,
        is_audio,
        is_video,
        merge_adjacent_segments,
        read_file,
        remove_short_segments,
    )


def test_no_legacy_imports() -> None:
    """The ``indextts.`` and ``clipperX.`` legacy namespaces must be gone."""
    import mediaflow

    for module in list(pkgutil.walk_packages(mediaflow.__path__, prefix="mediaflow.")):
        assert "indextts" not in module.name, f"legacy {module.name}"
        assert "clipperX" not in module.name, f"legacy {module.name}"
