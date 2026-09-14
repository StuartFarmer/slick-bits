# UniPrompt / Task Facet Learning

`UniPrompt(task, provider, errors, evaluate).run(initial, training, ...)` learns
task facets through grouped minibatch feedback and conceptual section edits.
`errors(instruction, batch)` returns serialized failures with expected/observed
outputs; `evaluate(instruction)` returns held-out quality, higher is better.
Both callbacks are async. Generation uses local Slick templates; configure
`TEMPLATE_ROOT` to `uniprompt/prompts` once at startup.

The loop periodically clusters per-example feedback, summarizes each group's
minibatch suggestions, sometimes forces add/delete/set exploration, edits every
beam member, and retains the best parents and children. Returned history contains
groups, beam snapshots and measured edit deltas. Generated partitions are checked;
empty text and nonfinite scores fail immediately. Errors propagate; no retries.

Inspected [official Microsoft UniPrompt](https://github.com/microsoft/UniPrompt/tree/1fdd3af8236e86f6b44619797bb90c4c2000d086):
`train.py`, `feedback/feedback.py`, `grouping/group.py`, `beam_search/beam.py`,
`metaprompts/default.yaml`. Task-specific answer extraction becomes the caller's
error callback; conceptual operations remain local. Group summary + individual
assignment calls are combined into one checked JSON partition. Group count is a
generation target. Histories reset when groups change to avoid misattribution.
The release accidentally leaves aggregate feedback empty on the non-exploration
path and never fills the shown edit history; this port aggregates that path and
records actual validation deltas. It initializes from one supplied prompt, as
the official beam implementation does. No external dataset/cache infrastructure
or experimental performance claims are included.
