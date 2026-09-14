"""Slick dialogue for proposing and revising the CMSA construction function."""

import argparse
import ast
import asyncio
import inspect
import json
from pathlib import Path

from pydantic import BaseModel, Field
from slick import prompt, prompts
from slick.providers import CodexCLI, OpenAIAPI, Provider

import cmsa as algorithm


class Proposal(BaseModel):
    rationale: str = Field(min_length=1)
    code: str = Field(min_length=1)
    checks: list[str] = Field(min_length=1)


# This standalone baseline is embedded into the complete CMSA source for prompting.
# Published V1/V2 formulas are deliberately absent from a fresh discovery dialogue.
BASELINE_FUNCTION = '''def generate_solution(neighbors, age, rng, *, variant="cmsa",
        determinism_rate=0.8, candidate_list_size=5, deadline=math.inf, order=None):
    """Construct using static minimum degree and a restricted candidate list."""
    active = list(order) if order is not None else sorted(
        range(len(neighbors)), key=lambda v: (len(neighbors[v]), v))
    solution = set()
    while active and time.perf_counter() < deadline:
        vertex = active[0] if rng.random() <= determinism_rate else rng.choice(
            active[:candidate_list_size])
        solution.add(vertex)
        if age[vertex] == -1:
            age[vertex] = 0
        active = [v for v in active if v != vertex and v not in neighbors[vertex]]
    return solution
'''


def baseline_source():
    tree = ast.parse(inspect.getsource(algorithm))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef) or node.name != "selection_probabilities"
    ]
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_solution":
            tree.body[i] = ast.parse(BASELINE_FUNCTION).body[0]
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "VARIANTS" for t in node.targets
        ):
            tree.body[i] = ast.parse('VARIANTS = ("cmsa",)').body[0]
    return ast.unparse(tree) + "\n"


@prompt(template="propose.j2", output_type=Proposal)
async def propose(
    source: str, mode: str, feedback: str, history: list[dict], *, generated: Proposal
) -> Proposal:
    """Propose construction code; the dialogue records subsequent interface validation."""
    return generated


def validate_candidate(code):
    """Syntax/interface check only; never executes model output and is not a sandbox."""
    tree = ast.parse(code)
    compile(tree, "<candidate>", "exec")
    if any(
        not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body
    ):
        raise ValueError("candidate may contain only imports and function definitions")
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_solution"
    ]
    if len(functions) != 1 or functions[0].decorator_list:
        raise ValueError("define exactly one undecorated generate_solution function")
    fn = functions[0]
    positional = fn.args.posonlyargs + fn.args.args
    if [arg.arg for arg in positional[:3]] != ["neighbors", "age", "rng"]:
        raise ValueError("first three parameters must be neighbors, age, rng")
    parameters = []
    for i, arg in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if i < len(fn.args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = inspect.Parameter.empty if i < len(positional) - len(fn.args.defaults) else None
        parameters.append(inspect.Parameter(arg.arg, kind, default=default))
    if fn.args.vararg:
        parameters.append(inspect.Parameter(fn.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        parameters.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None,
            )
        )
    if fn.args.kwarg:
        parameters.append(inspect.Parameter(fn.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    try:
        inspect.Signature(parameters).bind(
            None,
            None,
            None,
            variant="cmsa",
            determinism_rate=0.8,
            candidate_list_size=5,
            deadline=1.0,
            order=[],
        )
    except TypeError as exc:
        raise ValueError(f"candidate signature is incompatible: {exc}") from exc


class DemoProvider(Provider):
    """Canned V1 proposal; exercises Slick without calling an LLM."""

    async def acall(self, context, *, tools=None, tool_results=None):
        self.context = context
        proposal = Proposal(
            rationale="Offline demonstration: reuse the implemented paper V1 constructor.",
            code="""def generate_solution(neighbors, age, rng, *, variant="cmsa",
        determinism_rate=0.8, candidate_list_size=5, deadline=float("inf"), order=None):
    from cmsa import generate_solution as paper_constructor
    return paper_constructor(neighbors, age, rng, variant="v1",
        determinism_rate=determinism_rate, candidate_list_size=candidate_list_size,
        deadline=deadline, order=order)
""",
            checks=[
                "Verify independence and compare against the V1 constructor using identical seeds."
            ],
        )
        return proposal.model_dump_json(), []


async def dialogue(args):
    path = args.output / "dialogue.json"
    if path.exists():
        state = json.loads(path.read_text())
        if args.source and args.source.read_text() != state["source"]:
            raise ValueError("source changed; start a new output directory")
    else:
        state = {
            "source": args.source.read_text() if args.source else baseline_source(),
            "turns": [],
        }
    if args.provider == "demo":
        provider = DemoProvider()
    elif args.provider == "codex":
        provider = CodexCLI(model=args.model, timeout=args.timeout)
    else:
        provider = OpenAIAPI(model=args.model, timeout=args.timeout, max_output_tokens=8192)
    args.output.mkdir(parents=True, exist_ok=True)
    number = len(state["turns"]) + 1
    context = await propose.render(state["source"], args.mode, args.feedback, state["turns"])
    (args.output / f"prompt-{number:03d}.txt").write_text(context)
    result = await propose(
        state["source"], args.mode, args.feedback, state["turns"], provider=provider
    )
    validation = "Syntax and interface accepted; behavior has not been tested."
    try:
        validate_candidate(result.code)
    except (SyntaxError, ValueError) as exc:
        validation = f"REJECTED: {exc}"
    state["turns"].append(
        {
            "mode": args.mode,
            "feedback": args.feedback,
            "provider": args.provider,
            "model": args.model,
            "proposal": result.model_dump(),
            "validation": validation,
        }
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)
    candidate_path = args.output / f"candidate-{number:03d}.py"
    candidate_path.write_text(result.code)
    print(f"{candidate_path}\n{result.rationale}\n{validation}")
    if validation.startswith("REJECTED:"):
        raise ValueError("invalid candidate saved for feedback; revise before benchmarking")


def main(argv=None):
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("demo", "codex", "openai"), default="demo")
    p.add_argument("--model", help="required for openai; codex uses its configured default")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--mode", choices=("heuristic", "performance"), default="heuristic")
    p.add_argument("--feedback", default="Improve the construction heuristic.")
    p.add_argument(
        "--source",
        type=Path,
        help="complete Python CMSA source; defaults to baseline only",
    )
    p.add_argument("--output", type=Path, default=Path("runs/dialogue"))
    args = p.parse_args(argv)
    if args.provider == "openai" and not args.model:
        p.error("--model is required for openai")
    if args.provider == "demo" and args.model:
        p.error("--model requires a real provider")
    if args.timeout < 1:
        p.error("--timeout must be positive")
    try:
        asyncio.run(dialogue(args))
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
