# Persuasive debate implementation plan

Implement the supplied Khan et al. paper as a problem-agnostic Slick agent,
using the official UCL DARK implementation at commit
`f9c71d16e08bac1d30757511c5c7d231f34b4afc` as the algorithm reference.

- Build `agent.py` around a caller-supplied question, two answer strings, private
  evidence, debater providers, judge, and async token-logprob callback. Keep
  prompts local and select opening, challenge, rebuttal, question response,
  critique, and revision operations in Python.
- Preserve simultaneous snapshots, egocentric expert transcripts, best-of-N,
  critique selection, normalized quote matching, bounded rejection sampling,
  and answer-order judging. Include consultancy and interactive debate.
- Add independent `evaluation.py` functions for assignment-balanced matches,
  Swiss pairings, and least-squares Elo with the installed SciPy optimizer.
- Verify with `tests/test_persuasive_debate.py`, using only the shared
  `tests.providers.ScriptedProvider`: source isolation, snapshot semantics,
  candidate/scoring budgets, quote spoofing, fallback, swapped judgments,
  failure records, template rendering, and synthetic tournament ratings.
- Document startup, scoring integration, provenance, license, and deliberate
  differences. No QuALITY loader, human UI, training infrastructure, or paid
  experiments are needed for this algorithm library.

Caller configuration is trusted. Model output and measured score checks belong
at runtime boundaries. Every generation is independent; no mutable Session is
shared. Raw responses and failures remain inspectable after a failed run.
