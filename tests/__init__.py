"""Test suite for the MediaFlow refactor.

These tests cover the structural integrity of the unified package:
- importability of every public subpackage and module
- basic behaviour of shared helpers in :mod:`mediaflow.common`
- argument parser wiring for the unified CLI
- smoke checks for the clipper's translator cleanup lifecycle

Tests do not require GPU, network access, or heavy ML models.
"""
