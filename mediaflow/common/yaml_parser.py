"""YAML configuration loader with attribute-style access.

Adapted from the original ``clipperX/common/yaml_parser.py``. The main
improvements are:

* ``yaml.load`` was replaced with ``yaml.safe_load`` to silence the
  ``YAMLLoadWarning`` introduced in PyYAML 5.1+.
* The recursive ``strip`` and ``merge`` operations no longer mutate the
  internal ``__dict__`` of :class:`AttrDict` via reflection, which is fragile
  under CPython's data-model changes.
"""
from __future__ import annotations

import copy
import os
from ast import literal_eval
from fractions import Fraction
from typing import Any

import yaml

_HERE = os.path.dirname(__file__)


class AttrDict(dict):
    """A dict subclass that supports attribute-style access and deep merging."""

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self:
            value = self[name]
            if isinstance(value, AttrDict) or not isinstance(value, dict):
                return value
            promoted = AttrDict.cast(value)
            self[name] = promoted
            return promoted
        if name.startswith("__"):
            raise AttributeError(name)
        new = AttrDict()
        self[name] = new
        return new

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def __str__(self) -> str:
        return yaml.safe_dump(self.strip(), allow_unicode=True, default_flow_style=False)

    def merge(self, other: dict | "AttrDict") -> None:
        if not isinstance(other, AttrDict):
            other = AttrDict.cast(other)
        for key, value in other.items():
            value = copy.deepcopy(value)
            if key not in self or not isinstance(value, AttrDict):
                self[key] = value
                continue
            self[key].merge(value)

    def strip(self) -> Any:
        if not isinstance(self, dict):
            if isinstance(self, (list, tuple)):
                return str(tuple(self))
            return self
        return {k: AttrDict.strip(v) if isinstance(v, AttrDict) else v for k, v in self.items()}

    @staticmethod
    def cast(d: Any) -> Any:
        if not isinstance(d, dict):
            return d
        return AttrDict({k: AttrDict.cast(v) for k, v in d.items()})


def _coerce(value: Any) -> Any:
    """Best-effort conversion of YAML scalar strings to native Python types."""
    if not isinstance(value, str):
        return value
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        pass
    return value


def parse(node: Any) -> Any:
    if not isinstance(node, dict):
        if isinstance(node, str):
            return _coerce(node)
        return node
    return AttrDict({k: parse(v) for k, v in node.items()})


def load(fname: str) -> AttrDict:
    with open(fname, "r", encoding="utf-8") as fh:
        return parse(yaml.safe_load(fh))


class YamlParser(AttrDict):
    """High-level configuration loader that supports layered overrides."""

    def __init__(self, cfg_name: str = "config", path: str = ""):
        super().__init__()
        if path:
            self.add_cfg(path)
        if cfg_name:
            self.add_cfg(cfg_name)

    def add_args(self, args: Any) -> "YamlParser":
        if hasattr(args, "__dict__"):
            self.merge(vars(args))
        elif isinstance(args, dict):
            self.merge(args)
        return self

    def add_cfg(self, cfg: str, args: Any = None, update: bool = False) -> "YamlParser":
        if os.path.isfile(cfg):
            fname = cfg
            cfg_name = os.path.splitext(os.path.basename(cfg))[0]
        else:
            fname = os.path.join(_HERE, "../configs", cfg + ".yaml")
            cfg_name = cfg
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"YAML config not found: {fname}")
        self.merge(load(fname))
        self["name"] = cfg_name
        if args is not None:
            self.add_args(args)
        if cfg and args and update:
            self.save_cfg(fname)
        return self

    def save_cfg(self, fname: str) -> None:
        with open(fname, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.strip(), fh, allow_unicode=True, default_flow_style=False)

    def getdir(self) -> str:
        if "name" not in self:
            self["name"] = "testing"
        return os.path.join(self.ckpt_dir, self.name)

    def makedir(self) -> str:
        checkpoint_dir = self.getdir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        cfg_path = os.path.join(checkpoint_dir, "cfg.yaml")
        self.save_cfg(cfg_path)
        return checkpoint_dir
