"""Stochastic IDS / SIEM / Behavior Monitor.

Why this lives on the ENVIRONMENT side: a real sensor reacts to what actually
happened. So the simulator, which knows whether a request really was malicious,
asks each sensor to fire with its true-positive rate (malicious request) or its
false-positive rate (benign request). The controller only receives the
resulting True/False alarms, never the reason.

Simplifying assumption (state it in the report): the three sensors are treated
as conditionally independent given the truth, so their evidence is multiplied.
Real sensors that watch the same traffic are correlated.
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

from .live_config import SENSORS


class SensorSuite:
    """Fires each sensor independently for one request."""

    def __init__(self, rng: random.Random):
        self.rng = rng  # the simulation's seeded RNG, so alarms are reproducible

    def observe(self, request_is_malicious: bool, rng: random.Random = None) -> Dict[str, bool]:
        """Return {'IDS': bool, 'SIEM': bool, 'BM': bool} for one request.
        `rng` is the per-request RNG, so twin worlds get identical alarms for the same request."""
        return self.observe_detailed(request_is_malicious, rng)[0]

    def observe_detailed(self, request_is_malicious: bool, rng: random.Random = None) -> Tuple[Dict[str, bool], Dict[str, dict]]:
        """Same draws as observe(), plus how each alarm was produced: {sensor: {'p': rate used, 'u': random draw}}.
        The second part is ENVIRONMENT-ONLY (it depends on the truth) and is never given to a controller."""
        rng = rng or self.rng
        alarms, draws = {}, {}
        for name, (true_positive, false_positive) in SENSORS.items():
            fire_probability = true_positive if request_is_malicious else false_positive
            u = rng.random()
            alarms[name] = u < fire_probability
            draws[name] = {'p': fire_probability, 'u': u}
        return alarms, draws
