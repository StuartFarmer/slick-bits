# Problem-agnostic agent flows

The user authorizes replacing the problem-specific applications with eight
optimizer classes following the supplied PaperPlanner shape. Each owns its
task/provider/evaluator dependencies, decorated prompt methods, and an explicit
async run loop. Local prompt folders remain inside their implementation directory.

The archived applications preserve the original domains and infrastructure.
Active agents operate on caller-supplied textual candidates and finite scores;
QUBE additionally receives a behavior signature, and APEX receives embeddings.
No generic agent runs code, starts Docker, reads benchmark data, constructs a
provider, or contains a demo provider. Tests share tests.providers.ScriptedProvider.

All agents export from their package and live in agent.py. There is no shared
agent base class or CLI framework. Callers configure Slick's application template
root before a run; simultaneous independent template roots require separate
processes with the checked Slick API. Optional supplied Sessions follow the
reference's call-time execution pattern. New structured contracts and generic
instructions deliberately replace the earlier problem-specific prompt protocols.

- [x] Agent A: AEL, EoH, Optimizer; tests/test_ael.py, test_eoh.py, test_optimizer.py.
- [x] Agent B: APEX, EvoPROMPT; tests/test_apex.py, test_evoprompt.py.
- [x] Agent C: ReEvo, QUBE; tests/test_reevo.py, test_qube.py.
- [x] Parent: LLM_GP, shared test provider, archive, documentation, integration.
- [x] Verify algorithm-specific decisions, two unrelated task contexts, finite
  score rejection, failure accounting, local templates, and full offline tests.

Scout is an independent OpenAlex application, not an optimizer; its behavior is
outside this optimizer rewrite. Historical runs and local environments are not
migrated. Population selection, reflection, bandit, and island decisions remain
algorithm-specific; benchmark harnesses and traditional non-agent baselines stay
in the archive.
