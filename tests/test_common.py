"""Tests for :mod:`mediaflow.common` helpers."""
from __future__ import annotations

import os
import textwrap

import pytest

from mediaflow.common import (
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
    str_to_bool,
)


def test_attr_dict_attribute_access() -> None:
    from mediaflow.common import AttrDict

    cfg = AttrDict()
    cfg.foo.bar = 1
    assert cfg.foo.bar == 1
    assert isinstance(cfg.foo, AttrDict)


def test_yaml_parser_round_trip(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            name: demo
            values:
              - 1
              - 2
            """
        ),
        encoding="utf-8",
    )
    parser = YamlParser(cfg_name="", path=str(yaml_path))
    assert parser["name"] == "cfg"
    assert parser["values"] == [1, 2]


def test_md_task_lifecycle(tmp_path) -> None:
    md_path = tmp_path / "list.md"
    md = MD(str(md_path), encoding="utf-8")
    md.add_done_editing(False)
    md.add_task(True, "first sentence")
    md.add_task(False, "second sentence")
    md.write()

    md2 = MD(str(md_path), encoding="utf-8")
    marks = [m for m, _ in md2.tasks()]
    assert marks == [False, True, False]
    assert md2.done_editing() is False


def test_md_done_editing(tmp_path) -> None:
    md_path = tmp_path / "list.md"
    md = MD(str(md_path), encoding="utf-8")
    md.add_done_editing(True)
    md.write()

    md2 = MD(str(md_path), encoding="utf-8")
    assert md2.done_editing() is True


def test_expand_segments() -> None:
    segs = [{"start": 10.0, "end": 12.0}, {"start": 20.0, "end": 22.0}]
    expanded = expand_segments(segs, expand_head=1.0, expand_tail=1.0, total_length=30.0)
    assert expanded[0] == {"start": 9.0, "end": 13.0}
    assert expanded[1] == {"start": 19.0, "end": 23.0}


def test_remove_short_segments() -> None:
    segs = [{"start": 0.0, "end": 0.5}, {"start": 1.0, "end": 5.0}]
    out = remove_short_segments(segs, threshold=1.0)
    assert out == [{"start": 1.0, "end": 5.0}]


def test_merge_adjacent_segments() -> None:
    segs = [
        {"start": 0.0, "end": 2.0},
        {"start": 2.2, "end": 3.0},
        {"start": 10.0, "end": 12.0},
    ]
    merged = merge_adjacent_segments(segs, threshold=0.5)
    assert merged == [{"start": 0.0, "end": 3.0}, {"start": 10.0, "end": 12.0}]


def test_change_ext() -> None:
    assert change_ext("foo.srt", "md") == "foo.md"
    assert change_ext("foo", ".txt") == "foo.txt"


def test_add_cut() -> None:
    assert add_cut("foo.mp4") == "foo_cut.mp4"
    assert add_cut("foo_cut.mp4") == "foo__cut.mp4"


def test_is_video_audio() -> None:
    assert is_video("a.mp4") is True
    assert is_video("a.mp3") is False
    assert is_audio("a.mp3") is True
    assert is_audio("a.mp4") is False


def test_str_to_bool() -> None:
    assert str_to_bool("true") is True
    assert str_to_bool("false") is False
    with pytest.raises(ValueError):
        str_to_bool("yes")


def test_check_exists(tmp_path) -> None:
    f = tmp_path / "x.txt"
    assert check_exists(str(f), force=False) is False
    f.write_text("hi", encoding="utf-8")
    assert check_exists(str(f), force=False) is True
    assert check_exists(str(f), force=True) is False


def test_read_file(tmp_path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    assert read_file(str(f)) == "hello"


def test_logging_factory_creates_logger(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEDIAFLOW_LOG_DIR", str(tmp_path))
    logging_obj = Logging("mediaflow.test", log_cate="pytest")
    logger = logging_obj.get_logger()
    assert logger is not None
    assert os.path.exists(logging_obj.log_dir)
