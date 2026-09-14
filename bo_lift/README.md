# BO-LIFT

Implements the in-context Bayesian optimization loop from **Bayesian Optimization
of Catalysts With In-context Learning**, [paper](https://arxiv.org/abs/2304.05341).
The linked official [`ur-whitelab/bo-lift`](https://github.com/ur-whitelab/bo-lift)
now redirects to [BO-ICL](https://github.com/ur-whitelab/BO-ICL). Inspected
`boicl/asktell.py`, `llm_model.py`, and `aqfxns.py`: few-shot numeric prediction,
inverse candidate filtering, distribution construction, and discrete acquisition.

The class predicts numerical objective samples for each candidate from selected
measured examples, constructs their empirical distribution, and applies UCB,
expected improvement or probability of improvement. Optional inverse prompting
first asks for an ideal candidate at a perturbed aspirational outcome, then uses
embedding similarity plus maximal marginal relevance (MMR) to reduce the finite
candidate pool. New observations update subsequent contexts. Fewer than two
observations uses the source's random initialization rule.

```python
from pathlib import Path
from slick import prompts
from bo_lift import BOLIFT, Observation

prompts.TEMPLATE_ROOT = Path("/absolute/path/to/bo_lift/prompts")
agent = BOLIFT(task, provider, evaluate, embed)
result = await agent.run(candidate_descriptions, initial_observations,
                         acquisition="ei", inverse_count=16)
```

`evaluate(description)` is async and **higher-is-better**. `embed(descriptions)`
is async and returns one embedding row per description. The optimizer computes
cosine relevance, MMR, empirical moments and acquisition values itself. Initial
observations are `Observation(candidate, value)` objects. Already measured
candidates and duplicate pool strings are removed. `inverse_count=0` disables
inverse prompting; with `random_count>0` it chooses a random subset. The returned
best always comes from actual observations rather than the numerical surrogate.

This is the source's `use_logprobs=False` discrete-sampling path: each provider
completion contributes equal mass. It does not manufacture token probabilities
or ask the model to invent uncertainty. Optional logprob-weighted sampling,
Gaussian recalibration, quantile transforms and fine-tuned regressors are omitted.
Example selection uses MMR; `diversity=1` recovers pure cosine relevance. Inverse
examples use nearest measured objective values, and the desired outcome is
`best * Normal(1.2, 0.05)`, matching the source scaling. For signed objectives,
callers may need a positive shifted metric for this multiplicative target to be
aspirational. Python random sampling and task-neutral templates replace the
original provider/LangChain/FAISS infrastructure.

Numeric responses end in `###`; malformed or nonfinite predictions raise instead
of the source's unsafe index-shifting behavior after dropping empty predictions.
All provider/evaluation errors propagate. Counts include actual evaluations,
optimizer completions and embedding calls. Dependencies: Slick and NumPy.
Tests verify empirical EI, inverse filtering, uncertainty-sensitive selection,
and MMR diversity without model weights, paid calls or benchmark claims.
