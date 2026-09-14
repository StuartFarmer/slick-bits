# ExpeL implementation plan

Implement the supplied ExpeL Algorithms 1–3 and Section 4.4 as a generic Slick
agent, using the official LeapLabTHU/ExpeL implementation as the source reference.

- Add one owning class in `agent.py`, external operation templates in `prompts/`,
  and public exports. Inject async environment reset/step and task embeddings.
- Test the training → reflection → extraction → retrieval → evaluation flow with
  `tests/providers.py` before implementation. Cover terminal failures, retry and
  step budgets, all voting operators, malformed output, and held-out isolation.
- Preserve the paper's tagged insight operations and vote counts. Use explicit
  structured Slick output for ReAct actions, plain text for reflection/transfer.
- Use exact inner-product retrieval over successful task embeddings. Keep model,
  environment, embedding infrastructure, persistence, and benchmarks caller-owned.
- Document the official source commit, license, prompt/algorithm adaptations,
  startup template configuration, usage, and offline verification limits.
- Run the focused tests, neighboring Reflexion checks, lint, format, and template
  checks from another launch directory. Review the resulting files before finishing.
