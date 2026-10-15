"""Per-domain integration modules (registry + feasibility checkers + eval suites).

Each domain is a self-contained module under this package; importing the
module registers a :class:`DomainSpec` via :func:`registry.register_domain`.
The trainer / search / eval code dispatches through
:mod:`reap.rl.av_star.sim` which looks domains up by PDDL domain name.
"""
