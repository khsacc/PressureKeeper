"""Static end-of-run PNG report: sample pressure vs time, membrane (gas)
pressure vs time, and gas-vs-sample pressure, each with setpoint and actual
-- the same three panels as gui/live_plot.py's live pyqtgraph view, but
written once to the run directory so a run leaves behind a static artifact
(e.g. for a paper figure or reviewing a run after the fact) without needing
the GUI reopened. Reads ticks.csv back from disk rather than buffering rows
in memory as the run progresses, so the one function here works identically
whether called from logging_sink.py's DataLogger.close() or re-run later,
standalone, against any past run directory:

    python -m pressurekeeper.report_plots <run_dir>

matplotlib is an optional dependency (see pyproject.toml's `plotting`
extra), imported only here and only by DataLogger.close()'s lazy import, so
a run's CSV/manifest audit trail never depends on it being installed. Uses
the Agg canvas directly rather than pyplot: DataLogger.close() can run
inside the GUI process where PyQt6 already owns the Qt event loop, and
pyplot's own backend autodetection has no business anywhere near that.

Sample-side "setpoint": the sample-vs-time panel plots `user_target_gpa`
(the operator's final target, matching live_plot.py's own choice). The
gas-vs-sample panel plots `control_target_gpa` instead -- the per-tick value
the controller is actually steering toward right now, i.e. the same
"currently commanded" sense as `membrane_setpoint_mpa` -- since pairing a
moving gas setpoint with a near-constant final target would collapse that
trace to a flat, uninformative line.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_SAMPLE_COLOR = "#3b82f6"
_TARGET_COLOR = "#94a3b8"
_MEMBRANE_ACTUAL_COLOR = "#a855f7"

_TICK_COLUMNS = (
    "t_mono", "filtered_pressure_gpa", "user_target_gpa", "control_target_gpa",
    "membrane_setpoint_mpa", "membrane_actual_mpa",
)


def _read_ticks(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "ticks.csv"
    data: dict[str, list[float]] = {c: [] for c in _TICK_COLUMNS}
    if not path.exists():
        return data
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in _TICK_COLUMNS:
                raw = row.get(c, "")
                data[c].append(float(raw) if raw not in ("", None) else math.nan)
    return data


def write_summary_plots(run_dir: Path) -> Path | None:
    """Read `run_dir/ticks.csv` and write `run_dir/summary_plots.png`.
    Returns None (writes nothing) if ticks.csv is missing or has no rows,
    e.g. a run that was opened and closed without ever reaching a tick.
    """
    run_dir = Path(run_dir)
    data = _read_ticks(run_dir)
    if not data["t_mono"]:
        return None

    t0 = data["t_mono"][0]
    t = [x - t0 for x in data["t_mono"]]
    sample = data["filtered_pressure_gpa"]
    user_target = data["user_target_gpa"]
    control_target = data["control_target_gpa"]
    membrane_set = data["membrane_setpoint_mpa"]
    membrane_act = data["membrane_actual_mpa"]

    fig = Figure(figsize=(18, 5))
    FigureCanvasAgg(fig)
    ax_sample, ax_membrane, ax_corr = fig.subplots(1, 3)

    ax_sample.plot(t, user_target, color=_TARGET_COLOR, lw=1, ls="--", label="target")
    ax_sample.plot(t, sample, color=_SAMPLE_COLOR, lw=1.5, label="filtered")
    ax_sample.set_title("Sample pressure (ruby)")
    ax_sample.set_xlabel("elapsed time (s)")
    ax_sample.set_ylabel("Sample pressure (GPa)")
    ax_sample.legend()
    ax_sample.grid(alpha=0.2)

    ax_membrane.plot(t, membrane_set, color=_TARGET_COLOR, lw=1, label="setpoint")
    ax_membrane.plot(t, membrane_act, color=_MEMBRANE_ACTUAL_COLOR, lw=1.5, label="actual")
    ax_membrane.set_title("Membrane (gas) pressure")
    ax_membrane.set_xlabel("elapsed time (s)")
    ax_membrane.set_ylabel("Gas pressure (MPa)")
    ax_membrane.legend()
    ax_membrane.grid(alpha=0.2)

    ax_corr.plot(membrane_set, control_target, color=_TARGET_COLOR, lw=1, ls="--", label="setpoint")
    ax_corr.plot(membrane_act, sample, color=_SAMPLE_COLOR, lw=1, label="actual")
    ax_corr.set_title("Sample vs Membrane pressure")
    ax_corr.set_xlabel("Membrane pressure (MPa)")
    ax_corr.set_ylabel("Sample pressure (GPa)")
    ax_corr.legend()
    ax_corr.grid(alpha=0.2)

    fig.tight_layout()
    out_path = run_dir / "summary_plots.png"
    fig.savefig(out_path, dpi=150)
    return out_path


def _main(argv: list[str] | None = None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m pressurekeeper.report_plots <run_dir>", file=sys.stderr)
        return 2
    out = write_summary_plots(Path(argv[0]))
    if out is None:
        print("no ticks recorded in this run -- nothing to plot", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
