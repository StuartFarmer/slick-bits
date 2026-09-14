"""Assignment-balanced cross-play, Swiss matches, and least-squares Elo.

Adapted from UCL DARK's llm_debate (MIT); no task labels or model construction.
"""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

Play = Callable[[str, str], Awaitable[float]]


@dataclass(frozen=True)
class Match:
    first: str
    second: str
    win_rate: float


@dataclass(frozen=True)
class Tournament:
    ranking: tuple[str, ...]
    scores: dict[str, float]
    matches: tuple[Match, ...]
    byes: tuple[tuple[int, str], ...]


async def balanced_match(first: str, second: str, play: Play) -> Match:
    """play(x,y) returns x's vote share on a fixed question set and assignment.

    Reverse debater assignments using the same questions, then average. The
    callback owns model runs and answer-order judgments. No ground truth enters.
    """
    forward = await play(first, second)
    reverse = await play(second, first)
    for rate in (forward, reverse):
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Observed win rate must be finite and between zero and one")
    return Match(first, second, (forward + 1 - reverse) / 2)


async def swiss_tournament(
    players: Sequence[str],
    play: Play,
    *,
    rounds: int | None = None,
) -> Tournament:
    """Seeded nearest-neighbor pairing without rematches (Algorithm 2).

    Default rounds=ceil(log2(n)); each non-bye match runs both assignments.
    Ties earn half a point each; byes earn one and do not enter Elo fitting.
    """
    order = list(players)
    seed = {player: index for index, player in enumerate(order)}
    scores = dict.fromkeys(order, 0.0)
    played: set[frozenset[str]] = set()
    matches, byes = [], []
    rounds = math.ceil(math.log2(len(order))) if rounds is None else rounds
    for round_number in range(1, rounds + 1):
        unpaired = sorted(order, key=lambda player: (-scores[player], seed[player]))
        while unpaired:
            first = unpaired.pop(0)
            # ponytail: upstream greedy pairing can create extra byes; use a matching
            # solver if maximum-cardinality pairings become an experiment requirement.
            second = next((p for p in unpaired if frozenset((first, p)) not in played), None)
            if second is None:
                byes.append((round_number, first))
                scores[first] += 1
                continue
            unpaired.remove(second)
            match = await balanced_match(first, second, play)
            matches.append(match)
            played.add(frozenset((first, second)))
            scores[first] += 0.5 if match.win_rate == 0.5 else float(match.win_rate > 0.5)
            scores[second] += 0.5 if match.win_rate == 0.5 else float(match.win_rate < 0.5)
    ranking = sorted(order, key=lambda player: (-scores[player], seed[player]))
    return Tournament(tuple(ranking), scores, tuple(matches), tuple(byes))


def fit_elo(matches: Sequence[Match], *, reference: str) -> dict[str, float]:
    """Fit Appendix D.5's unweighted squared win-rate error using BFGS.

    SciPy is required only for this function. The reference is fixed at zero.
    The match graph must connect every rated player to the reference; otherwise
    the observations cannot determine relative ratings and fitting fails.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    players = list(dict.fromkeys(p for match in matches for p in (match.first, match.second)))
    neighbors = {p: set() for p in players}
    for match in matches:
        neighbors[match.first].add(match.second)
        neighbors[match.second].add(match.first)
    reached, pending = {reference}, [reference]
    while pending:
        for player in neighbors[pending.pop()] - reached:
            reached.add(player)
            pending.append(player)
    if reached != set(players):
        raise ValueError("Disconnected match graph cannot determine relative Elo ratings")
    free = [p for p in players if p != reference]
    indices = {p: i for i, p in enumerate(free)}

    def objective(strengths):
        ratings = {reference: 0.0, **dict(zip(free, strengths))}
        loss, gradient = 0.0, np.zeros(len(free))
        for match in matches:
            predicted = expit(math.log(10) * (ratings[match.first] - ratings[match.second]))
            error = predicted - match.win_rate
            loss += error**2
            derivative = 2 * error * math.log(10) * predicted * (1 - predicted)
            for player, sign in ((match.first, 1), (match.second, -1)):
                if player != reference:
                    gradient[indices[player]] += sign * derivative
        return loss, gradient

    result = minimize(
        objective, np.zeros(len(free)), jac=True, method="BFGS", options={"gtol": 1e-9}
    )
    if not result.success:
        raise RuntimeError(f"Elo fit did not converge: {result.message}")
    return {reference: 0.0, **{p: float(400 * result.x[i]) for p, i in indices.items()}}
