"""Bundle the paper's task families without owning retrieval infrastructure."""

from pathlib import Path

from pydantic import BaseModel, JsonValue
from slick import prompt
from slick.providers import Provider

from .agent import Decomp, Handler, Program


class Answer(BaseModel, extra="forbid"):
    answer: JsonValue


class Handlers:
    """Each sub-task has independent examples and a replaceable prompt method."""

    @prompt(template="split.j2", output_type=Answer)
    async def split(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="arr_position.j2", output_type=Answer)
    async def arr_position(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="merge.j2", output_type=Answer)
    async def merge(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="list_split.j2", output_type=Answer)
    async def list_split(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="reverse_base.j2", output_type=Answer)
    async def reverse_base(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="qa.j2", output_type=Answer)
    async def qa(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="singlehop_qa.j2", output_type=Answer)
    async def singlehop_qa(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="multihop_qa.j2", output_type=Answer)
    async def multihop_qa(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="cot.j2", output_type=Answer)
    async def cot(self, question: str, *, generated: Answer) -> Answer:
        return generated

    @prompt(template="gpt_ans.j2", output_type=Answer)
    async def gpt_ans(self, question: str, *, generated: Answer) -> Answer:
        return generated


def paper_agent(
    provider: Provider,
    *,
    context: str = "",
    retrieve: Handler | None = None,
    handlers: dict[str, Handler] | None = None,
) -> Decomp:
    """Construct bundled programs; caller configures Slick's template root once.

    retrieve(query) returns JSON-compatible documents containing titles and text.
    Pass handler overrides to substitute symbolic functions or other models.
    """
    library = Handlers()
    library.context = context

    def bind(method):
        async def answer(question):
            return (await method(question, provider=provider)).answer

        return answer

    names = (
        "split",
        "arr_position",
        "merge",
        "list_split",
        "reverse_base",
        "qa",
        "singlehop_qa",
        "multihop_qa",
        "cot",
        "gpt_ans",
    )
    callbacks = {name: bind(getattr(library, name)) for name in names}
    if retrieve is not None:
        callbacks["retrieve"] = retrieve
    callbacks.update(handlers or {})
    root = Path(__file__).resolve().parent / "prompts/programs"
    programs = {path.stem: Program(path.read_text()) for path in sorted(root.glob("*.txt"))}
    if "retrieve" not in callbacks:
        del programs["retrieve_odqa"]
        del programs["open_qa"]
    return Decomp(
        "Solve the question using the specialized sub-task handlers and prompting programs.",
        provider,
        callbacks,
        programs,
    )
