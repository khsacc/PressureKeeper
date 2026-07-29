"""PyQt6 GUI entry point. Same CLI surface as `pressurekeeper.cli` (--config,
--sim, --target, --dry-run/--live) so the two front ends are launched the
same way; this one opens a window instead of taking over the terminal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from ..app import build_app
from ..config import load_config
from ..instance_lock import InstanceAlreadyRunning
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

    # Built before build_app() so InstanceAlreadyRunning (which build_app()
    # can raise before any window exists) can still be shown as a dialog
    # instead of only a terminal traceback the operator may not be watching.
    app = QApplication([sys.argv[0]] + qt_argv)
    try:
        ctx = build_app(config, use_simulator=args.sim, dry_run=args.dry_run, sim_physics=sim_physics)
    except InstanceAlreadyRunning as e:
        QMessageBox.critical(None, "PressureKeeper already running", str(e))
        return 1
    if args.target is not None:
        ctx.controller.set_target(args.target)

    # ctx.controller.config, not the `config` loaded above: build_app() can
    # override control.dry_run from --dry-run/--live (see its own dry_run
    # parameter) without that override ever being reflected back into this
    # function's `config` variable. Passing the original here used to leave
    # MainWindow's "Configure Parameters" dialog prefilled from a dry_run
    # value that could disagree with what the controller actually believed
    # -- e.g. showing "dry_run: checked" while a --live launch was actually
    # driving the real membrane.
    effective_config = ctx.controller.config
    loop_interval_s = max(effective_config.ruby_api.poll_interval_s, effective_config.control.loop_min_interval_s)
    window = MainWindow(ctx, config=effective_config, config_path=args.config, poll_interval_s=loop_interval_s)
    window.setWindowTitle(window.windowTitle() + f" — {'SIMULATOR' if args.sim else ('DRY-RUN' if effective_config.control.dry_run else 'LIVE')}")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
