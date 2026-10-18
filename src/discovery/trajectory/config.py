"""Tiny helper: use one section of a ``configs/discovery/trajectory_*.yaml`` file as argparse defaults."""
from __future__ import annotations

import argparse


def parse_args_with_config(parser: argparse.ArgumentParser, section: str, required: tuple[str, ...] = ()):
    """Parse CLI args; values from ``--config <yaml>`` section ``section`` act as defaults.

    Explicit command-line flags always win. Keys must match argparse ``dest`` names.
    """
    parser.add_argument("--config", default=None, help=f"YAML config; its '{section}' section provides defaults")
    pre, _ = parser.parse_known_args()
    if pre.config:
        import yaml

        with open(pre.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        values = dict(cfg.get(section) or {})
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(values) - known)
        if unknown:
            raise KeyError(f"Unknown keys in section '{section}' of {pre.config}: {unknown}")
        parser.set_defaults(**values)
    args = parser.parse_args()
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + m for m in missing))
    return args
