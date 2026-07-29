# PressureKeeper

One-sided, predictive, stepwise pressure control for a membrane-driven
diamond anvil cell (mDAC).

- **Actuator**: Druck PACE5000 controlling He membrane pressure (max 6 MPa),
  accessed via the HTTP API of a separate existing PACE5000 control app, [PaceMaker](https://github.com/khsacc/PaceMaker)
  (started with its `--api` flag). This project never talks to the PACE5000
  hardware directly — it only calls that app's API, which owns the serial
  link and device-level safety checks.
- **Feedback**: sample pressure from ruby fluorescence, read via the HTTP API
  of a separate spectroscopy control app, [FluoRaPressée](https://github.com/khsacc/FluoraPressee), at up to ~4 Hz.

Because pressure generation in DACs is largely irreversible, active pressure
control **only ever raises actual membrane pressure**. The sole setpoint
decrease is a safe rebase while the output is already in Measure; it cannot
drive the membrane downward. This is not a PID loop — it is a
predictive, one-directional, stepwise state machine that prioritizes
avoiding overshoot over convergence speed.

## Why not PID

A PID controller assumes it can correct in both directions. Here, pushing
past the target cannot be undone by normal control (the membrane doesn't
release), and the plant gain itself grows with pressure, so a fixed-gain
loop that behaves well at low pressure will overshoot badly at high
pressure. Instead, each step is sized from a conservative, pressure-dependent
sensitivity estimate, and no new step is issued until the previous one has
demonstrably finished responding.

## Architecture

```
pressurekeeper/
  models.py        data classes: samples, GainRegion, StepRecord, snapshots, control states
  config.py         pydantic configuration models + YAML/TOML loader
  clock.py           Clock protocol (MonotonicClock for real use, FakeClock for tests/sim time)
  interfaces.py       RubyPressureSource / MembranePressureController protocols
  estimator.py         PressureEstimator: outlier-suppressed filtering, slope, validity
  gain.py               GainEstimator: online, pressure-binned sensitivity estimation
  safety.py              SafetySupervisor: the only thing allowed to veto pressurization
  controller.py           OneSidedPressureController: the state machine
  logging_sink.py          DataLogger: CSV time series (ticks / commands / steps / events)
  clients/
    pace5000_client.py      HTTP client for the PACE5000 control app's API
    ruby_client.py            HTTP client for the ruby-pressure control app's API
  sim/
    simulator.py               offline physics simulator (same Protocols as real clients)
  app.py             wiring: Configuration -> fully assembled controller (real or simulated)
  cli.py              interactive terminal UI
```

Device I/O is fully abstracted behind two `Protocol`s
(`RubyPressureSource`, `MembranePressureController`); the real HTTP clients
and the simulator are interchangeable implementations of the same interface,
and the controller/estimator/safety/gain code never imports `requests` or
the simulator directly.

## Control loop

Each tick (paced by the ruby API's ~4 Hz ceiling):

1. Read ruby pressure (`RubyPressureSource.read()`), feed `PressureEstimator`.
2. Read PACE5000 status (`MembranePressureController.read_status()`).
3. `SafetySupervisor.evaluate(...)` — if it says `pause` or `abort`, no
   pressurization logic runs this tick.
4. Otherwise, `OneSidedPressureController._advance()`:
   - compute `sizing_pressure = max(filtered, latest_raw)` and
     `predicted_pressure = max(filtered + max(slope, 0) * prediction_horizon_s,
     sizing_pressure)` (an upward raw reading can only reduce pressurization;
     filter lag or a downward outlier can never justify a larger step);
   - if `sizing_pressure > target + overshoot_margin` → **HOLD** (never push further);
   - if `predicted >= target - reach_margin` → **HOLD** (close enough, wait);
   - if in HOLD and `filtered < target - reapproach_margin` → back to **APPROACH**
     (asymmetric hysteresis: small dips during HOLD are ignored, only a
     larger fall-back re-triggers pressurization);
   - otherwise, if the previous step hasn't settled yet → **SETTLE** (wait);
   - once settled, compute and issue the next membrane step (see below), or
     do nothing this tick if there is nothing meaningful to command.

Startup and every recovery from Measure (Reset, PAUSE recovery, HOLD
re-approach, or an externally disabled Control state) use the same guarded
sequence:

1. confirm that PACE5000 is in Measure;
2. discard interrupted response/gain tracking;
3. calculate one safe step from the fresh **actual membrane pressure**, not
   the possibly stale device target;
4. write that target while still in Measure;
5. obtain a later, fresh status confirming both the written target and
   `positive supply pressure > proposed setpoint`;
6. only then enable Control.

Thus an old target cannot move the membrane briefly between Reset/resume and
the application of a newly calculated target.

A step counts as **settled** only after both (a) at least
`GainRegion.minimum_settle_time_s` has elapsed since the command (a hard
blackout — no new command is ever issued sooner), and (b)
`abs(pressure_slope) < settled_slope_threshold_gpa_s` has held continuously
for that same duration. Near the target (within `near_target_distance_gpa`),
the slope threshold is tightened and the wait is extended
(`near_target_slope_threshold_scale`, `near_target_extra_settle_time_s`).

**Tuning note learned while testing this against the simulator:**
`minimum_settle_time_s` must clear the system's real dead time with margin.
If it doesn't, "no response has arrived yet" (flat because the dead-time
delay hasn't elapsed) is indistinguishable from "settled" under a pure
slope-threshold criterion, and the controller can stack multiple oversized
commands before any of them has shown its real effect. `config/default.yaml`
uses 8-25 s specifically to stay well clear of any plausible dead time;
don't shrink these without characterizing your actual system's dead time
first.

### Step sizing

```
control_target = user_target - approach_margin
predicted_error = control_target - predicted_pressure
requested_sample_step = min(approach_factor * predicted_error, max_sample_step_for_region)
safe_gain = GainEstimator.estimate(...)   # conservative, pressure-binned, online
membrane_step = clamp(requested_sample_step / safe_gain, 0, region.max_membrane_step)
```

`safe_gain` starts from a conservative, per-region prior
(`GainRegion.safe_gain`, from config) and switches to an online estimate
(median + `safety_factor * spread`, floored at the configured upper
percentile of observed gains) once enough settled steps have been observed
near the current pressure — always biased toward the safe (larger) side,
never the average.

### States

`APPROACH` → issuing/waiting to issue the next step · `SETTLE` → waiting out
a step already issued · `HOLD` → at/near target, doing nothing ·
`PAUSE` → safety-vetoed or operator-requested, auto-clears if the trigger
was automatic and non-manual · `ABORT` → sticky, requires an explicit
operator `reset` (never auto-clears).

**PAUSE/HOLD/ABORT all actively stop the membrane ("STOP"), not just
withhold new commands.** The PACE5000 ramps toward whatever setpoint/rate it
was last sent, on its own, independent of this controller — merely blocking
*new* steps leaves any step already in flight (and, on this hardware, its
nonlinear high-pressure gain) free to keep pushing the sample past a target
or past a just-detected fault. Instead, entering `ABORT`, `PAUSE` (manual or
safety-triggered), or `HOLD` (target reached, overshoot margin exceeded, or
the target lowered mid-ramp — the overshoot check re-runs every tick
regardless of prior state) switches the PACE5000 out of control mode into
measure-only (`Pace5000Client.set_control_mode(False)`, `_stop_membrane`),
which halts the device's own drive toward its setpoint immediately,
regardless of what that setpoint is or how stale the last sample reading is.
This is retried every tick until a status read-back confirms control mode is
actually off — a failed write is never a one-shot "aborted but still
ramping" latch. Leaving `PAUSE`/`HOLD` back into `APPROACH` does not directly
re-arm the old target; it executes the guarded Measure → safe target →
readback → Control sequence above. `reset` likewise remains in Measure until
that sequence has completed.

**One exception: a `PAUSE` episode caused, every tick since it began, solely
by the compression-rate cap** (`compression_rate_exceeded`) does not sit
inertly in Measure for its whole duration. Once Measure is confirmed,
`_try_hold_at_current_pressure` stages the *fresh actual* membrane pressure
(no forward step, unlike the guarded sequence's "+ one safe step") as the new
setpoint and re-enables Control, so the PACE5000's own regulation counters
small leaks while the observed slope is still above the cap, instead of
leaving the sample to drift unchecked for however long the slope takes to
decay. This can retreat the setpoint below the just-interrupted in-flight
target — safe (never below actual pressure) but a deliberate, bounded
exception to "setpoint never decreases," justified because the one-sided
design's premise against reversing course is about the predictor's model
being direction-asymmetric, not a hard safety wall. It is intentionally
narrow: any other event on top (a `hard_sample_jump`, a comm-error streak
within the slope window, a manual pause, ...) disqualifies the whole episode,
and it never fires close enough to target that `HOLD`'s own hysteresis would
have handled it anyway. It only ever writes once per episode — Control, once
re-armed against a fixed setpoint, keeps holding it without further writes.

Lowering a target while a step is in flight immediately enters `HOLD` and
STOPs the output. Because normal control is one-sided and STOP does not lower
the old device setpoint, the controller will not re-arm that old setpoint for
a smaller replacement target; the operator must restore at least the
pre-reduction target (or make an explicit recovery decision).

An operator's `abort()`/`pause()` additionally issues this stop command
immediately, in the calling thread, independent of `self._lock` and of
whatever the polling loop's current tick is doing — so a comm failure that's
itself the reason for aborting can never delay the abort. `quit`/Ctrl-C/GUI
window-close also stop-and-confirm before the process exits
(`controller.stop_and_confirm()`).

## Safety

`SafetySupervisor` is the only thing that can block a write. It watches:
absolute sample/membrane pressure limits (including a dangerous PACE5000
*target* left over from before this controller started polling, not just
actual pressure), missing/non-finite status fields, per-command and
cumulative (sliding-window) membrane step limits, an operator-configured
compression-rate cap checked against the *observed* slope (not just used to
size requested steps), stale/invalid/jumpy/non-finite ruby readings, ruby
and PACE5000 communication errors (time-based, from the first error in a
streak — not reset by every subsequent error), commanded-vs-reported
setpoint mismatches (zero-grace by default), an excessive active
setpoint/actual gap, missing or insufficient positive supply pressure, and
manual pause/abort requests. While Control is active, supply pressure must
continuously exceed the target; immediately before every setpoint write, a
fresh status from that same loop tick must also show supply pressure above
the proposed setpoint. Any `pause`/`abort`
condition blocks new pressurization; `abort` is sticky until
`force_reset()` (exposed via `controller.reset()`, an explicit operator
action — never called automatically). Automatic de-pressurization is out of
scope. A recovery may lower a stale *setpoint* only after Measure is
confirmed, rebasing it to `fresh actual + one safe step`; actual membrane
pressure is never intentionally lowered. `set_control_mode(False)` remains
the STOP action that freezes an in-flight increase.

A write that raises after possibly having already applied on the device
(HTTP response lost, not a rejection) is treated as *possibly issued*, not
"never sent" — the cumulative step budget and pending-step bookkeeping
account for it as if it succeeded, and the commanded-vs-reported
`setpoint_mismatch` check is what reconciles reality on the next status
read-back, rather than the controller blindly stacking another command on
top of an unknown real state.

## Configuration

See `config/default.yaml` (documented inline). All device-specific numbers
— API URLs/keys, gain schedule, safety limits, hysteresis margins — live
there; nothing device-specific is hardcoded. Fields marked `SITE-SPECIFIC`
must be reviewed before use, in particular:

- `ruby_api.base_url` / `ruby_api.api_key` (the ruby control app's host and
  the API key it issues)
- `ruby_api.acquisition.*` (fitting/exposure parameters, ruby zero-pressure
  peak position, pressure scale)
- `safety.max_sample_pressure_gpa` (your experiment's planned ceiling)
- `safety.minimum_source_pressure_headroom_mpa` (strict minimum is zero,
  meaning supply must still be greater than setpoint)
- `gain_regions` (calibrate the gain schedule against your actual cell)

The default independent sample compression-rate ceiling is
`0.5 GPa/min`. It can be changed at runtime from the GUI or cleared
explicitly, but should only be loosened after reviewing the cell response.

## Running

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Or, equivalently, run `./setup.sh` (macOS/Linux) or `setup.bat` (Windows) from
the repo root — either creates `.venv` and installs the same `[dev]` extras
(add `--gui` for the GUI extras). After that, every command below is a single
invocation of something under `.venv/`.

**Simulator** (no hardware, no network — safe to run anytime):

```bash
.venv/bin/pressurekeeper --config config/default.yaml --sim --target 1.0
```

**Against real APIs, rehearsal mode** (reads real ruby data, never writes to
PACE5000 — the default; `control.dry_run: true` in config, or force with
`--dry-run`):

```bash
.venv/bin/pressurekeeper --config config/default.yaml --target 1.0
```

Dry-run remains usable when the real PACE5000 is in measure-only mode: the
state machine simulates control-mode intent internally so it can log would-be
steps, but never toggles the real output or sends a pressure setpoint. Opening
the GUI likewise does not enable PACE5000 control mode; live output is first
reconciled only after the operator starts the loop and a safety tick has
evaluated both devices.

**Live** (actually moves the membrane — requires explicit opt-in):

```bash
.venv/bin/pressurekeeper --config config/default.yaml --target 1.0 --live
```

Interactive commands while running: `target <GPa>`, `pause`, `resume`,
`abort`, `reset`, `status`, `quit`. The status line refreshes in place twice
a second showing state, filtered/predicted pressure, slope, membrane
setpoint/actual, positive supply pressure, the currently-used safe gain, and
any active safety reasons. A logging failure is shown as `LOGGING-ERROR` but
does not itself PAUSE pressure control.

### Running the GUI

Needs the `gui` extras installed (`.venv/bin/pip install -e ".[dev,gui]"`, or
`./setup.sh --gui` / `setup.bat --gui`). Same flags as the CLI above
(`--config` required; `--sim`, `--seed`, `--target`, `--dry-run`/`--live`),
via the `pressurekeeper-gui` console script:

```bash
.venv/bin/pressurekeeper-gui --config config/default.yaml --sim --target 1.0
```

The window title shows which mode is active (`SIMULATOR` / `DRY-RUN` /
`LIVE`). A live plot and the Pause/Resume/Abort/Reset controls are always
visible above two tabs, **Single Target** and **Schedule** (the latter
disables the former while a multi-target run is in progress). Opening the
window does not start anything — sensors are first read and the PACE5000 is
first reconciled only after clicking **Start Control**. Closing the window
(or Ctrl-C) stops and confirms the PACE5000 output before exiting, the same
as `quit` in the CLI.

## Logging

Each run writes to `logging.directory/<run_name>/` (auto-timestamped with
microsecond precision by default; an explicitly reused name is rejected
rather than truncating an earlier audit trail): `ticks.csv` (one row per
control iteration), `commands.csv` (one
row per PACE5000 write, with the reason and every decision value that
produced it — predicted pressure, predicted error, gain used, etc.),
`steps.csv` (one row per settled step: before/after pressures, response
time, observed gain — the online sensitivity data set), and `events.csv`
(safety events and state transitions).

Logging initialization and runtime write failures are sticky and visible in
GUI/CUI status. They do not enter PAUSE, because filesystem availability is
not a pressure safety signal; if initialization fails, the run continues
with logging disabled and an explicit warning.

## Testing

```bash
.venv/bin/pytest -q
```

`tests/test_estimator.py`, `test_gain.py`, `test_safety.py`, `test_config.py`,
`test_pace5000_client.py` unit-test each component in isolation (including
non-finite/missing-field fail-closed behavior, the ruby-API request-body
contract, and PACE5000 `control_mode` wire-format coercion).
`test_controller_unit.py` tests the state machine with fully scripted,
deterministic inputs, including STOP-on-PAUSE/HOLD, abort-stop retry after a
failed write, startup/Reset stale-setpoint rebasing before Control, supply
pressure interlocking, a visible logging failure never causing PAUSE or
blocking a stop, and an ambiguous
(comm-error) write not being stacked on top of. `test_scenarios.py` runs the
full closed loop against the simulator (nonlinear gain, lag, dead time,
creep, noise, outliers, irreversibility) for every scenario in the spec:
low- and high-pressure convergence, gain surge, outliers, a ruby API outage,
a slow PACE5000/sample response, creep near target, an underestimated-gain
surprise, approaching the hard limit (including a forced overshoot past it),
a small dip during HOLD, a manual stop mid-run, and — going beyond "no new
commands are issued" to actually confirm the membrane's *actual* pressure
stops climbing — ABORT/PAUSE/HOLD mid-ramp and a target lowered mid-ramp.
All tests run on a `FakeClock`, so wall-clock time is not a factor.

Schedules are also one-sided: a set-pressure step below the currently active
target is rejected as an error. This prevents a safety `HOLD` caused by target
reduction from being misreported as if the lower sample pressure had actually
been reached.
