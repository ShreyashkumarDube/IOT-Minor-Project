"""Twin worlds: the Bayesian and LLM engines run in two separate, identical simulations.

    same scenario + same seed
     ┌──────────┴──────────┐
  World B (Bayesian)    World L (LLM)
  own environment copy  own environment copy
  ZeroTrustController   LLMController
  own ledger, nonces,   own ledger, nonces,
  windows, log, policy  windows, log, policy
     └──────────┬──────────┘
  comparison layer (reads both, writes neither) + ground truth for scoring only

Nothing is shared between the worlds: each is built from its own deep copy of the
scenario. Because every random draw is derived from (seed, device, tick, stream,
request index), both worlds generate the same traffic and sensor alarms; only the
two controllers' decisions differ. Both worlds advance in lockstep, one tick at a time.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .controller import ZeroTrustController
from .live_sim import LiveSimulation, load_scenario, scenario_network
from .simulator import Device

MODES = {'bayesian': ('bayesian',), 'llm': ('llm',), 'both': ('bayesian', 'llm')}
MODE_LABELS = {'bayesian': 'Bayesian only', 'llm': 'LLM only', 'both': 'Both — isolated comparison'}


class TwinSimulation:
    """Holds one or two worlds (depending on the run mode) and steps them together."""

    def __init__(self, devices: Sequence[Device], connections: Sequence[Tuple[str, str]], scenario: Optional[dict],
                 seed: int, mode: str = 'bayesian', llm_controller_factory: Optional[Callable[[], object]] = None,
                 bayes_controller_factory: Callable[[], object] = ZeroTrustController):
        if mode not in MODES:
            raise ValueError(f'unknown mode {mode!r}')
        if 'llm' in MODES[mode] and llm_controller_factory is None:
            raise ValueError('an LLM controller factory is needed for LLM modes')
        self.mode, self.seed = mode, seed
        self.worlds: Dict[str, LiveSimulation] = {}
        for engine in MODES[mode]:      # the other world is not created at all in single-engine modes
            controller = bayes_controller_factory() if engine == 'bayesian' else llm_controller_factory()
            self.worlds[engine] = LiveSimulation(copy.deepcopy(list(devices)), list(connections),
                                                 scenario=copy.deepcopy(scenario), seed=seed, controller=controller)
        self.scenario_label = self.primary.scenario_label

    @classmethod
    def from_scenario(cls, name_or_path: str, seed: Optional[int] = None, mode: str = 'bayesian', **kwargs) -> 'TwinSimulation':
        scenario = load_scenario(name_or_path)
        devices, connections = scenario_network(scenario)
        return cls(devices, connections, scenario, scenario.get('seed', 0) if seed is None else seed, mode, **kwargs)

    # ----- accessors ---------------------------------------------------------------------
    @property
    def bayes(self) -> Optional[LiveSimulation]:
        return self.worlds.get('bayesian')

    @property
    def llm(self) -> Optional[LiveSimulation]:
        return self.worlds.get('llm')

    @property
    def primary(self) -> LiveSimulation:
        """The Bayesian world if it exists, otherwise the LLM world."""
        return self.bayes or self.llm

    @property
    def tick(self) -> int:
        return self.primary.tick

    # ----- running -----------------------------------------------------------------------------
    def step(self, progress=None) -> Dict[str, List[dict]]:
        """Advance every world by one tick (lockstep). Returns new records per engine."""
        return {engine: world.step(progress if engine == 'llm' else None) for engine, world in self.worlds.items()}

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def trigger_attack(self, device_id: str, agent_type: str) -> int:
        """Apply the same environment change to every world (controllers are not told)."""
        starts = {world.trigger_attack(device_id, agent_type) for world in self.worlds.values()}
        return starts.pop()

    def shutdown(self) -> None:
        for world in self.worlds.values():
            close = getattr(world.controller, 'shutdown', None)
            if close:
                close()
