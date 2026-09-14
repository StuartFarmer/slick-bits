# InstOptima

Implements [Yang and Li, 2023, Algorithm 1](https://arxiv.org/pdf/2310.17630):
four objective-guided definition/example mutation and crossover operators,
component-preserving offspring, non-dominated sorting, crowding selection, and
optional random Pareto-member refresh described in section 2.2.

Inspected official [NSGA-II code](https://github.com/yangheng95/InstOptima/blob/master/evo_core/nsga2.py)
and [instruction operators](https://github.com/yangheng95/InstOptima/blob/master/operators/instruction_operators.py).
That release uses unary definition operations despite accepting a second parent;
it also evolves operation prompts. This port follows the paper's four fixed
operators, not those release-specific differences.

Adaptations: generic tasks, JSON contracts, caller-owned evaluation instead of
fine-tuning infrastructure, normalized crowding, maximization-oriented scores,
and a configurable 0.1 refresh probability. Seeded parent selection follows the
paper's random second parent; scores are reused within a generation. Refresh
can remove elites; set its probability to zero for strict elitist survival.

```python
from pathlib import Path
from slick import prompts
from instoptima import InstOptima

prompts.TEMPLATE_ROOT = Path("instoptima/prompts").resolve()
# async evaluate(instruction) -> (accuracy, -length, -perplexity)
result = await InstOptima(task, provider, evaluate, objectives).run(initial_instructions)
```

Configure the process-global root once. Each objective needs one finite score;
higher is better. Errors propagate without retries. Results expose the population,
Pareto subset, and evaluation count. Tests verify search mechanics, not paper results.
