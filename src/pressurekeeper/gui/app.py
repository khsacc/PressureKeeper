"""PyQt6 GUI entry point. Same CLI surface as `pressurekeeper.cli` (--config,
--sim, --target, --dry-run/--live) so the two front ends are launched the
same way; this one opens a window instead of taking over the terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ..app import build_app
from ..config import load_config
from ..sim import DACPhysicsConfig
from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pressurekeeper-gui",
        description="PyQt6 GUI for the one-sided predictive pressure controller.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML or TOML configuration file")
    parser.add_argument("--target", type=float, default=None, help="Initial target sample pressure, in GPa")
    parser.add_argument("--sim", action="store_true", help="Use the built-in simulator instead of real devices")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --sim")
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument("--dry-run", dest="dry_run", action="store_const", const=True, default=None,
                                help="Force-disable real PACE5000 writes (overrides config)")
    dry_run_group.add_argument("--live", dest="dry_run", action="store_const", const=False,
                                help="Enable real PACE5000 writes — DANGEROUS: this moves the real membrane")
    args, qt_argv = parser.parse_known_args(argv)

    config = load_config(args.config)
    sim_physics = DACPhysicsConfig(seed=args.seed) if args.sim else None
    ctx = build_app(config, use_simulator=args.sim, dry_run=args.dry_run, sim_physics=sim_physics)
    if args.target is not None:
        ctx.controller.set_target(args.target)

    app = QApplication([sys.argv[0]] + qt_argv)
    loop_interval_s = max(config.ruby_api.poll_interval_s, config.control.loop_min_interval_s)
    window = MainWindow(ctx, poll_interval_s=loop_interval_s)
    window.setWindowTitle(window.windowTitle() + f" — {'SIMULATOR' if args.sim else ('DRY-RUN' if (args.dry_run if args.dry_run is not None else config.control.dry_run) else 'LIVE')}")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
