"""Generate a heuristic proposal and check its declared interface without executing it."""

import ast
import asyncio
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, StringConstraints, validate_call
from slick import prompt, prompts

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Proposal(BaseModel, extra="forbid"):
    thought: Text
    code: Text


def check_interface(code: str) -> None:
    """Check the declared signature; runtime binding, output behavior, and safety are unchecked."""
    tree = ast.parse(code)
    compile(tree, "<proposal>", "exec")
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "priority"
    ]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("define exactly one undecorated priority function")
    args = functions[0].args
    if (
        [arg.arg for arg in args.args] != ["item", "bins"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.defaults
        or args.vararg
        or args.kwarg
    ):
        raise ValueError("priority must accept exactly item and bins, without defaults")


class HeuristicDesigner:
    """Keep task context across proposals; the caller owns evaluation and retries."""

    @validate_call
    def __init__(self, task: Text):
        self.task = task

    @validate_call
    @prompt(template="heuristic/propose.j2", output_type=Proposal)
    async def propose(self, feedback: Text, *, generated: Proposal) -> Proposal:
        """Validate feedback before generation, then check the generated function interface."""
        check_interface(generated.code)
        return generated


class DemoProvider:
    """Return one canned proposal; no model reasoning or candidate execution."""

    async def acall(self, context):
        return Proposal(
            thought="Prefer the feasible bin with the least remaining capacity.",
            code="def priority(item, bins):\n    return [-capacity for capacity in bins]\n",
        ).model_dump_json(), []


async def main() -> None:
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    designer = HeuristicDesigner("Assign an item to a feasible bin; higher priority wins.")
    proposal = await designer.propose("Use a deterministic heuristic.", provider=DemoProvider())
    print(proposal.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
