"""Tiny markdown task list parser/editor used by the autocut workflow."""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple


class MD:
    """In-memory representation of a markdown task list file.

    The autocut workflow expects a file of the form:

    .. code-block:: markdown

        - [ ] [1,00:00] some subtitle text
        - [x] [2,00:03] another subtitle
        - [x] <-- Mark if you are done editing.

    The class only supports the limited grammar used by MediaFlow; it is not a
    full markdown implementation.
    """

    EDIT_DONE_MARK = "<-- Mark if you are done editing."

    def __init__(self, filename: str = "", encoding: str = "utf-8"):
        self.lines: List[str] = []
        self.encoding = encoding
        self.filename = filename
        if filename:
            self.load_file()

    def load_file(self) -> None:
        if os.path.exists(self.filename):
            with open(self.filename, encoding=self.encoding) as fh:
                self.lines = fh.readlines()

    def clear(self) -> None:
        self.lines = []

    def write(self) -> None:
        with open(self.filename, "wb") as fh:
            fh.write("\n".join(self.lines).encode(self.encoding, "replace"))

    def tasks(self) -> List[Tuple[Optional[bool], str]]:
        """Return ``[(is_marked, content), ...]`` for every task line."""
        ret: List[Tuple[Optional[bool], str]] = []
        for line in self.lines:
            mark, task = self._parse_task_status(line)
            if mark is not None:
                ret.append((mark, task))
        return ret

    def done_editing(self) -> bool:
        for mark, task in self.tasks():
            if mark and self.EDIT_DONE_MARK in task:
                return True
        return False

    def add(self, line: str) -> None:
        self.lines.append(line)

    def add_task(self, mark: bool, contents: str) -> None:
        self.add(f'- [{"x" if mark else " "}] {contents.strip()}')

    def add_done_editing(self, mark: bool) -> None:
        self.add_task(mark, self.EDIT_DONE_MARK)

    def add_video(self, video_fn: str) -> None:
        ext = os.path.splitext(video_fn)[1][1:]
        self.add(
            f'\n<video controls="true" allowfullscreen="true"> '
            f'<source src="{video_fn}" type="video/{ext}"> </video>\n'
        )

    @staticmethod
    def _parse_task_status(line: str) -> Tuple[Optional[bool], str]:
        match = re.match(r"- +\[([ x])\] +(.*)", line)
        if not match:
            return None, line
        return match.group(1).lower() == "x", match.group(2)
