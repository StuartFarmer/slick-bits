"""Text-based LLM_GP and tree GP for the paper's symbolic regression example."""

import ast
import copy
import math
import operator
import random
import time
from dataclasses import dataclass
from functools import lru_cache

VARIANTS = ("llm-gp-mu-xo", "llm-gp", "gp", "random", "llm-random")
TERMINALS = ("x0", "x1", "0", "1")
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
MAX_CHARS, MAX_NODES = 4096, 256


class BudgetExceeded(Exception):
    """Stop evolution while retaining already evaluated solutions."""


@dataclass
class Config:
    variant: str = "llm-gp-mu-xo"
    population_size: int = 10
    generations: int = 30
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    max_depth: int = 5
    seed: int = 0
    train_size: int | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        for name in ("population_size", "generations", "max_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_depth > 6:
            raise ValueError("max_depth must be <= 6 to respect expression limits")
        for rate in (self.crossover_rate, self.mutation_rate):
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("variation probabilities must be in [0, 1]")
        if self.train_size is not None and (
            type(self.train_size) is not int or self.train_size < 1
        ):
            raise ValueError("train_size must be a positive integer")


def parse_expression(expression):
    """Whitelist a bounded expression; no eval(), compile(), or Python execution."""
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_CHARS:
        raise ValueError("expression length outside limits")
    tree = ast.parse(expression.strip(), mode="eval").body
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ValueError("too many expression nodes")
    for node in nodes:
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            continue
        if isinstance(node, ast.Name) and node.id in ("x0", "x1"):
            continue
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1):
            continue
        if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult)):
            continue
        raise ValueError("only +, -, *, x0, x1, 0, 1 are allowed")
    return tree


def value(tree, x0, x1):
    if isinstance(tree, ast.Constant):
        return float(tree.value)
    if isinstance(tree, ast.Name):
        return x0 if tree.id == "x0" else x1
    return BINARY[type(tree.op)](value(tree.left, x0, x1), value(tree.right, x0, x1))


def evaluate(expression, data):
    """Mean squared error; invalid or nonfinite arithmetic has worst fitness."""
    try:
        tree = parse_expression(expression)
        score = math.fsum((value(tree, x0, x1) - y) ** 2 for x0, x1, y in data) / len(data)
        return score if math.isfinite(score) else math.inf
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return math.inf


def split_data(seed):
    """A reconstructed 121-point integer grid; split before subsampling training."""
    rows = [(float(a), float(b), float(a * a + b * b)) for a in range(-5, 6) for b in range(-5, 6)]
    random.Random(seed).shuffle(rows)
    return {"holdout": rows[:25], "test": rows[25:54], "train": rows[54:]}


def random_tree(rng, depth, full=False):
    if depth == 0 or (not full and rng.random() < 0.5):
        return parse_expression(rng.choice(TERMINALS))
    return ast.BinOp(
        left=random_tree(rng, depth - 1, full),
        op=rng.choice(tuple(BINARY))(),
        right=random_tree(rng, depth - 1, full),
    )


def tree_depth(tree):
    return (
        0
        if not isinstance(tree, ast.BinOp)
        else 1 + max(tree_depth(tree.left), tree_depth(tree.right))
    )


def subtree_paths(tree, path=()):
    yield path
    if isinstance(tree, ast.BinOp):
        yield from subtree_paths(tree.left, path + ("left",))
        yield from subtree_paths(tree.right, path + ("right",))


def subtree(tree, path):
    for field in path:
        tree = getattr(tree, field)
    return tree


def replace_subtree(tree, path, replacement):
    if not path:
        return copy.deepcopy(replacement)
    result = copy.deepcopy(tree)
    setattr(subtree(result, path[:-1]), path[-1], copy.deepcopy(replacement))
    return result


def tree_variation(parents, rng, config):
    trees = [parse_expression(p) for p in parents]
    if rng.random() < config.crossover_rate:
        paths = [rng.choice(list(subtree_paths(t))) for t in trees]
        children = [
            replace_subtree(trees[i], paths[i], subtree(trees[1 - i], paths[1 - i]))
            for i in range(2)
        ]
        trees = [c if tree_depth(c) <= config.max_depth else p for c, p in zip(children, trees)]
    for i, tree in enumerate(trees):
        if rng.random() < config.mutation_rate:
            path = rng.choice(list(subtree_paths(tree)))
            trees[i] = replace_subtree(tree, path, random_tree(rng, config.max_depth - len(path)))
    return [ast.unparse(t) for t in trees]


def fitness(individual):
    return math.inf if individual["fitness"] is None else individual["fitness"]


async def evolve(config, data, operators=None, on_generation=None):
    """Count every evaluation request, including cached expressions and elites."""
    rng = random.Random(config.seed)
    use_llm = config.variant.startswith("llm")
    if use_llm and operators is None:
        raise ValueError("LLM variants require Operators")
    for name in ("train", "test", "holdout"):
        rows = data[name]
        if not rows or any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in rows):
            raise ValueError(f"{name} must contain finite (x0, x1, target) rows")
    train_size = config.train_size or (10 if use_llm else len(data["train"]))
    if train_size > len(data["train"]):
        raise ValueError("train_size exceeds training split")
    training = random.Random(config.seed).sample(list(data["train"]), train_size)
    history, population, pending = [], [], []
    evaluations = 0
    best = None
    started = time.perf_counter()
    stop_reason = "generations"

    @lru_cache(maxsize=4096)
    def score(expression):
        return evaluate(expression, training)

    def assess(expression):
        nonlocal evaluations, best
        result = score(expression)
        evaluations += 1
        individual = {
            "expression": expression,
            "fitness": result if math.isfinite(result) else None,
        }
        if best is None or fitness(individual) < fitness(best):
            best = individual
        return individual

    def snapshot(generation, complete=True):
        record = {
            "generation": generation,
            "complete": complete,
            "evaluations": evaluations,
            "seconds": time.perf_counter() - started,
            "best_fitness": best["fitness"] if best else None,
            "mean_size": sum(len("".join(p["expression"].split())) for p in population)
            / len(population),
            "invalid_fitness": sum(p["fitness"] is None for p in population),
            "population": population[:],
        }
        history.append(record)
        if on_generation:
            on_generation(record)

    try:
        for generation in range(config.generations):
            pending = []
            if generation == 0 or config.variant in ("random", "llm-random"):
                for i in range(config.population_size):
                    expression = (
                        await operators.initialize()
                        if use_llm
                        else ast.unparse(
                            random_tree(rng, 1 + (i // 2) % config.max_depth, full=i % 2 == 0)
                        )
                    )
                    pending.append(assess(expression))
            else:
                if config.variant in ("gp", "llm-gp-mu-xo"):
                    pending.append(assess(min(population, key=fitness)["expression"]))
                while len(pending) < config.population_size:
                    if config.variant == "llm-gp":
                        parents = await operators.choose("selection", population, 2)
                    else:
                        parents = [min(rng.choices(population, k=2), key=fitness) for _ in range(2)]
                    expressions = [p["expression"] for p in parents]
                    if use_llm:
                        samples = [p["expression"] for p in population]
                        children = (
                            await operators.crossover(expressions, samples)
                            if (rng.random() < config.crossover_rate)
                            else expressions
                        )
                        for child in children[: config.population_size - len(pending)]:
                            if rng.random() < config.mutation_rate:
                                child = await operators.mutate(child, samples)
                            pending.append(assess(child))
                    else:
                        children = tree_variation(expressions, rng, config)
                        pending.extend(
                            assess(c) for c in children[: config.population_size - len(pending)]
                        )
                if config.variant == "llm-gp":
                    pending = await operators.choose(
                        "replacement", population + pending, config.population_size
                    )
            population = pending
            snapshot(generation)
    except BudgetExceeded:
        stop_reason = "budget"
        if pending:
            population = pending
            snapshot(len(history), complete=False)

    designated = best
    if config.variant == "llm-gp" and population and stop_reason != "budget":
        try:
            designated = (await operators.choose("best", population, 1))[0]
        except BudgetExceeded:
            stop_reason = "budget"
    result = {
        "stop_reason": stop_reason,
        "evaluations": evaluations,
        "unique_evaluations": score.cache_info().misses,
        "seconds": time.perf_counter() - started,
        "best": best,
        "designated_best": designated,
        "training_data": training,
        "history": history,
    }
    # Test and holdout are only consulted after all evolutionary decisions.
    for label, candidate in (("best", best), ("designated", designated)):
        for split in ("test", "holdout"):
            error = evaluate(candidate["expression"], data[split]) if candidate else math.inf
            result[f"{label}_{split}_mse"] = error if math.isfinite(error) else None
    return result
