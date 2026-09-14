"""Classical typed subtree mutation over a caller-defined primitive set."""

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class Primitive:
    name: str
    returns: str
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tree:
    primitive: Primitive
    children: tuple["Tree", ...] = ()

    def render(self) -> str:
        if not self.children:
            return self.primitive.name
        return f"{self.primitive.name}({', '.join(child.render() for child in self.children)})"


def subtree_mutate(
    tree: Tree,
    primitives: Sequence[Primitive],
    *,
    rng: random.Random,
    max_depth: int = 4,
) -> Tree:
    """Uniformly select a node, then grow a type-compatible replacement subtree.

    Root depth is zero. Supply a well-typed tree within max_depth and a grammar
    admitting replacements. Trees are immutable; unchanged mutations are valid.
    Symbol choice is uniform among primitives feasible at the remaining depth.
    """
    paths = []

    def visit(node, path):
        paths.append((path, node))
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    @lru_cache(None)
    def choices(returns, remaining):
        return tuple(
            primitive
            for primitive in primitives
            if primitive.returns == returns
            and (
                not primitive.arguments
                or remaining > 0
                and all(choices(argument, remaining - 1) for argument in primitive.arguments)
            )
        )

    def grow(returns, remaining):
        primitive = rng.choice(choices(returns, remaining))
        return Tree(primitive, tuple(grow(arg, remaining - 1) for arg in primitive.arguments))

    def substitute(node, path, replacement):
        if not path:
            return replacement
        children = list(node.children)
        children[path[0]] = substitute(children[path[0]], path[1:], replacement)
        return replace(node, children=tuple(children))

    visit(tree, ())
    path, target = rng.choice(paths)
    replacement = grow(target.primitive.returns, max_depth - len(path))
    return substitute(tree, path, replacement)
