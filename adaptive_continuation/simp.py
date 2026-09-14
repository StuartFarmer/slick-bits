"""SIMP continuation constants and schedules, without geometry or solver dependencies."""

from .agent import Gate, Measurement, Parameter

PARAMETERS = {
    "penal": Parameter(1, 5, 1, "SIMP penalization exponent"),
    "beta": Parameter(1, 64, 1, "Heaviside projection sharpness"),
    "rmin": Parameter(1.1, 4, 1.5, "Density filter radius", monotone="decrease"),
    "move": Parameter(0.03, 0.4, 0.2, "Optimality Criteria move limit"),
}
GATES = (Gate("grayness", "beta", 0.20, 8, threshold_key="grayness_gate"),)
TAIL = {"penal": 4.5, "beta": 32, "rmin": 1.20, "move": 0.05}
SETTINGS = {
    "grayness_gate": 0.20,
    "call_every": 5,
    "penal_ramp": 12,
    "beta_double": 10,
    "phase_penal": 22,
    "phase_sharp": 16,
}
TUNABLES = {
    "grayness_gate": Parameter(0.10, 0.35, 0.20, "Grayness threshold for beta cap"),
    "call_every": Parameter(3, 15, 5, "Iterations between model calls", integer=True),
    "penal_ramp": Parameter(4, 20, 12, "Fallback penalization ramp length", integer=True),
    "beta_double": Parameter(5, 20, 10, "Fallback beta doubling period", integer=True),
    "phase_penal": Parameter(4, 25, 22, "Minimum penalization stage length", integer=True),
    "phase_sharp": Parameter(4, 25, 16, "Minimum sharpening stage length", integer=True),
}


def valid_snapshot(measured: Measurement, parameters: dict[str, float]) -> bool:
    """The caller reports volume feasibility via Measurement.feasible."""
    return parameters["penal"] >= 3 and measured.metrics["grayness"] < 0.25


def grayness(densities) -> float:
    """Equation (7) for a flat collection of projected physical densities."""
    return 4 * sum(rho * (1 - rho) for rho in densities) / len(densities)


def schedule_only(iteration: int, budget: int, settings: dict) -> dict[str, float]:
    """Interpolate Table 1's advisory targets using only consumed budget."""
    fraction = iteration / budget
    if fraction < 0.08:
        return {"penal": 1 + fraction / 0.08, "beta": 1, "rmin": 1.5, "move": 0.2}
    if fraction < 0.5:
        progress = (fraction - 0.08) / 0.42
        return {"penal": 2 + 2.5 * progress, "beta": 1 + 3 * progress, "rmin": 1.35, "move": 0.15}
    if fraction < 0.75:
        return {"penal": 4.5, "beta": 4 + 12 * (fraction - 0.5) / 0.25, "rmin": 1.25, "move": 0.08}
    return TAIL.copy()


def fallback(iteration: int, budget: int, settings: dict) -> dict[str, float]:
    """Reconstruct the unspecified fallback using all six Table 3 settings.

    Stage minima are measured in solver iterations. Timing fields affect this
    fallback only; the call period and grayness gate affect the live controller.
    """
    exploration_end = 0.08 * budget
    penal_end = max(0.5 * budget, exploration_end + settings["phase_penal"])
    sharp_end = max(0.75 * budget, penal_end + settings["phase_sharp"])
    if iteration < exploration_end:
        return schedule_only(iteration, budget, settings)
    if iteration < penal_end:
        age = iteration - exploration_end
        return {
            "penal": 2 + 2.5 * min(1, age / settings["penal_ramp"]),
            "beta": 2 ** min(2, age / settings["beta_double"]),
            "rmin": 1.35,
            "move": 0.15,
        }
    if iteration < sharp_end:
        age = iteration - penal_end
        return {
            "penal": 4.5,
            "beta": 4 * 2 ** min(2, age / settings["beta_double"]),
            "rmin": 1.25,
            "move": 0.08,
        }
    return TAIL.copy()
