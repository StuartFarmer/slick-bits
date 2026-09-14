"""Appendix B's generator and reflector prompts, rendered by Slick."""

from slick import prompt


@prompt(template="generate.j2")
async def generate(
    task, stage, *, seed_code="", worse=None, better=None, reflection="", generated: str
) -> str:
    """Return candidate source; the caller owns interface checks and evaluation."""
    return generated


@prompt(template="reflect_pair.j2")
async def reflect_pair(task, worse, better, *, generated: str) -> str:
    """Return textual guidance comparing the supplied candidates."""
    return generated


@prompt(template="reflect_long.j2")
async def reflect_long(task, prior, insights, *, generated: str) -> str:
    """Return a reflection summary; the caller bounds carried memory."""
    return generated
