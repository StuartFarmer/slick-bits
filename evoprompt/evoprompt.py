"""EvoPROMPT GA and DE: Slick generates prompts; Python owns evolution and fitness."""

import math
import random
import re
from collections.abc import Awaitable, Callable, Sequence

from slick import prompt
from slick.providers import Provider


@prompt(template="ga_offspring.j2")
async def ga_offspring(parent1: str, parent2: str, *, generated: str) -> str:
    """Generate raw GA offspring; the optimizer logs it before extracting the prompt."""
    return generated


@prompt(template="de_offspring.j2")
async def de_offspring(donor1: str, donor2: str, best: str, target: str, *, generated: str) -> str:
    """Generate raw DE offspring from the fixed generation parents."""
    return generated


@prompt(template="variation.j2")
async def variation(instruction: str, *, generated: str) -> str:
    """Generate a tagged variation to fill the initial population."""
    return generated


def extract_prompt(response: str) -> str:
    """Reject ambiguous or missing final prompts instead of scoring the explanation."""
    match = re.search(r"<prompt>(.*?)</prompt>", response, flags=re.DOTALL)
    if (
        response.count("<prompt>") != 1
        or response.count("</prompt>") != 1
        or match is None
        or not match.group(1).strip()
    ):
        raise ValueError("expected exactly one nonempty <prompt>...</prompt> in the response")
    return match.group(1).strip()


async def optimize(
    initial_prompts: Sequence[str],
    evaluate: Callable[[str], Awaitable[float]],
    provider: Provider,
    *,
    algorithm: str = "de",
    population_size: int | None = None,
    iterations: int = 10,
    seed: int = 0,
    cache: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Maximize finite, nonnegative dev fitness; no test data enters this function.

    GA samples parents with replacement and retains the best N of 2N prompts.
    DE samples two distinct donors excluding the target, uses the generation's
    best as Prompt 3, and replaces each target only on strict improvement.
    Each generation reads a frozen population. Ties prefer incumbents.
    Missing initial members are variations of the supplied manual prompts.
    Cached fitness assumes repeatable evaluation; set cache=False for noisy runs.
    Provider, evaluator, and logging failures propagate without hidden retries.
    """
    if isinstance(initial_prompts, str) or not initial_prompts:
        raise ValueError("supply a nonempty sequence of initial prompts")
    if any(not isinstance(p, str) or not p.strip() for p in initial_prompts):
        raise ValueError("initial prompts must be nonempty strings")
    prompts = [p.strip() for p in initial_prompts]
    size = len(prompts) if population_size is None else population_size
    if algorithm not in ("ga", "de"):
        raise ValueError("algorithm must be ga or de")
    minimum = 2 if algorithm == "ga" else 3
    if type(size) is not int or size < minimum or size < len(prompts):
        raise ValueError(
            f"population_size must be >= {minimum} and >= the number of initial prompts"
        )
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")

    rng = random.Random(seed)
    fitness = {}
    counts = {"evaluations": 0, "cache_hits": 0, "optimizer_calls": 0}

    def emit(event):
        if on_event is not None:
            on_event(event)

    async def score(instruction):
        cached = cache and instruction in fitness
        if cached:
            counts["cache_hits"] += 1
            value = fitness[instruction]
        else:
            value = float(await evaluate(instruction))
            if not math.isfinite(value) or value < 0:
                raise ValueError("fitness must be finite and nonnegative (higher is better)")
            counts["evaluations"] += 1
            if cache:
                fitness[instruction] = value
        candidate = {"prompt": instruction, "score": value}
        emit({"event": "evaluation", **candidate, "cached": cached})
        return candidate

    async def generate(operator, parents, iteration, slot):
        response = await operator(*parents, provider=provider)
        counts["optimizer_calls"] += 1
        # Log before parsing so malformed model output remains inspectable.
        emit(
            {
                "event": "evolution",
                "operator": operator.__name__,
                "iteration": iteration,
                "slot": slot,
                "parents": parents,
                "response": response,
            }
        )
        return extract_prompt(response)

    manual = prompts.copy()
    while len(prompts) < size:
        prompts.append(await generate(variation, [rng.choice(manual)], 0, len(prompts)))
    population = [await score(p) for p in prompts]
    initial_best = max(population, key=lambda p: p["score"])
    history = []

    def snapshot(iteration):
        entry = {
            "iteration": iteration,
            "best_score": max(p["score"] for p in population),
            "mean_score": sum(p["score"] / size for p in population),
        }
        history.append(entry)
        emit({"event": "population", **entry, "population": population.copy()})

    snapshot(0)
    # ponytail: sequential calls bound API load; add bounded concurrency if latency matters.
    for iteration in range(1, iterations + 1):
        best = max(population, key=lambda p: p["score"])
        offspring = []
        if algorithm == "ga":
            # Scaling preserves roulette probabilities without overflowing their sum.
            maximum = best["score"]
            weights = [p["score"] / maximum for p in population] if maximum else None
            for slot in range(size):
                parents = [p["prompt"] for p in rng.choices(population, weights=weights, k=2)]
                child = await generate(ga_offspring, parents, iteration, slot)
                offspring.append(await score(child))
            population = sorted(population + offspring, key=lambda p: p["score"], reverse=True)[
                :size
            ]
        else:
            for slot, target in enumerate(population):
                donors = rng.sample([j for j in range(size) if j != slot], 2)
                parents = [population[j]["prompt"] for j in donors]
                parents.extend([best["prompt"], target["prompt"]])
                child = await generate(de_offspring, parents, iteration, slot)
                trial = await score(child)
                offspring.append(trial if trial["score"] > target["score"] else target)
            population = offspring
        snapshot(iteration)

    return {
        "algorithm": algorithm,
        "seed": seed,
        "iterations": iterations,
        "population_size": size,
        "cache": cache,
        "initial_best": initial_best,
        "best": max(population, key=lambda p: p["score"]),
        "population": population,
        "history": history,
        **counts,
    }
