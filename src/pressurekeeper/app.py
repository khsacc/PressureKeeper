"""Wiring: turn a `Configuration` into a fully assembled controller, choosing
real HTTP clients or the offline simulator behind the exact same interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .clients import Pace5000Client, RubyPressureClient
from .clock import Clock, MonotonicClock
from .config import Configuration
from .controller import OneSidedPressureController
from .gain import GainEstimator
from .estimator import PressureEstimator
from .instance_lock import SingleInstanceLock, lock_path_for
from .interfaces import MembranePressureController, RubyPressureSource
from .logging_sink import DataLogger
from .safety import SafetySupervisor
from .sim import DACPhysicsConfig, SimulatedDAC, SimulatedMembraneController, SimulatedRubySource


@dataclass
class AppContext:
    controller: OneSidedPressureController
    logger: DataLogger | None
    ruby: RubyPressureSource
    membrane: MembranePressureController
    clock: Clock
    dac: SimulatedDAC | None = None
    instance_lock: SingleInstanceLock | None = None

    def close(self) -> None:
        if self.logger is not None:
            try:
                self.logger.close()
            except Exception:
                pass
        for dev in (self.ruby, self.membrane):
            close = getattr(dev, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if self.instance_lock is not None:
            try:
                self.instance_lock.release()
            except Exception:
                pass


def build_app(
    config: Configuration,
    *,
    use_simulator: bool = False,
    dry_run: bool | None = None,
    clock: Clock | None = None,
    sim_physics: DACPhysicsConfig | None = None,
    lock_dir: str | Path | None = None,
) -> AppContext:
    if dry_run is None and use_simulator:
        # The simulator is itself the safety boundary — nothing physical can
        # move — so a config-file dry_run (meant to gate real PACE5000
        # writes) must not silently suppress writes to the simulated
        # membrane too. An explicit --dry-run/--live still wins over this.
        dry_run = False
    if dry_run is not None:
        config = config.model_copy(update={"control": config.control.model_copy(update={"dry_run": dry_run})})

    # Best-effort guard against a second PressureKeeper process on this same
    # machine talking to the same PACE5000 endpoint -- see instance_lock.py.
    # The simulator never touches a real endpoint, so it's exempt.
    instance_lock: SingleInstanceLock | None = None
    if not use_simulator:
        instance_lock = SingleInstanceLock(lock_path_for(config.pace5000_api.base_url, lock_dir))
        instance_lock.acquire()  # raises InstanceAlreadyRunning if another live instance holds it

    try:
        clock = clock or MonotonicClock()
        estimator = PressureEstimator(config.estimator)
        gain_estimator = GainEstimator(config.gain_estimation)
        safety = SafetySupervisor(config.safety, clock.now())
        logger: DataLogger | None
        logging_error: str | None = None
        mode = "simulator" if use_simulator else ("dry-run" if config.control.dry_run else "live")
        try:
            logger = DataLogger(config.logging, mode=mode)
        except Exception as e:
            # Loss of logging is visible but is not itself a reason to move or
            # stop pressure. This also lets an experiment recover its audit path
            # externally without making device safety depend on filesystem state.
            logger = None
            logging_error = f"{type(e).__name__}: {e}"

        dac: SimulatedDAC | None = None
        if use_simulator:
            dac = SimulatedDAC(sim_physics or DACPhysicsConfig(), clock.now())
            ruby: RubyPressureSource = SimulatedRubySource(dac, clock)
            membrane: MembranePressureController = SimulatedMembraneController(dac, clock)
        else:
            ruby = RubyPressureClient(config.ruby_api)
            pace = Pace5000Client(config.pace5000_api, dry_run=config.control.dry_run)
            membrane = pace

        controller = OneSidedPressureController(
            config, ruby, membrane, estimator, gain_estimator, safety,
            logger=logger, clock=clock, initial_logging_error=logging_error,
        )
    except Exception:
        if instance_lock is not None:
            instance_lock.release()
        raise
    return AppContext(controller=controller, logger=logger, ruby=ruby, membrane=membrane, clock=clock, dac=dac,
                       instance_lock=instance_lock)
