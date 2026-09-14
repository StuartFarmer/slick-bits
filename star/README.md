# STaR: Self-Taught Reasoner

Problem-agnostic Slick implementation of Algorithm 1 in
[STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465),
using the supplied paper and the authors' [official code](https://github.com/ezelikman/STaR).
The supplied [APET link](https://github.com/daankepel/APET) concerns a different
method, already implemented in [`../apet`](../apet/README.md).

Each round:

1. Generate one rationale and answer for every labeled input using the current model.
2. Check final answers with the caller's async evaluator.
3. Rationalize incorrect answers with the same model, supplying the correct answer
   as a hint. Check those answers too; discard failures.
4. Combine accepted direct and rationalized solutions, with **unhinted** training
   prompts, and fine-tune a fresh copy of the **original** base checkpoint.
5. Generate the next round's data using the newly trained model.

The training set is replaced each round. Demonstrations remain fixed and appear
in generation and training prompts; rationalization demonstrations additionally
show answer hints. `rationalization=False` implements Algorithm 2.

## Use with your task and training backend

Use `optimizer/.venv/bin/python`, which has the adjacent Slick checkout installed.
No new dependencies are needed. Set Slick's template root once at application
startup, before any calls:

```python
from pathlib import Path

from slick import prompts
import star

prompts.TEMPLATE_ROOT = Path(star.__file__).resolve().parent / "prompts"


async def bootstrap(task, base_provider, examples, demonstrations, train_from_base):
    async def correct(input, answer, expected):
        return answer == expected

    async def fine_tune(base, training, iteration):
        # Your backend loads a fresh copy of base and returns a Slick provider.
        # This is the paper's default exponential step schedule.
        return await train_from_base(
            base, training, steps=int(40 * 1.2 ** (iteration - 1))
        )

    agent = star.STaR(
        task, base_provider, correct, fine_tune, demonstrations=demonstrations
    )
    return await agent.run(examples, iterations=4)
```

Supply labeled data as `Example(input="...", answer="...")` and a small separate
seed set as `Demonstration(input="...", rationale="...", answer="...")`.
Inputs and answers are arbitrary text: include choices, documents, code, or other
context in the input, and define the answer format in `task`. Empty demonstration
sets are allowed, but the paper bootstraps from a few rationale examples.

The four constructor dependencies are:

| Argument | Contract |
| --- | --- |
| `task` | Shared task instructions; do not include evaluation targets here |
| `provider` | Stateless Slick provider for the original base checkpoint |
| `evaluate` | Async `(input, generated_answer, expected_answer) -> bool` |
| `fine_tune` | Async `(original_base_provider, training_rows, one_based_iteration) -> new_provider` |

The evaluator defines correctness, including normalization or multiple valid
answers. It receives no rationale and does not need to judge reasoning quality.
For execution-based checks, the evaluator owns execution isolation.

**A real fine-tuning callback is required.** Slick supplies generation, not weight
training. The callback must load the original weights each round, train on the
supplied rows, preserve the original checkpoint, and return a provider bound to
the new weights. Returning the same unchanged model cannot reproduce STaR's
learning. Keep backend-specific checkpoint handles in your provider or callback.
This follows the repository's existing injected weight-optimization convention.

Each immutable `TrainingExample` has:

- `example`: original labeled input.
- `prompt`: complete unhinted generation prompt, including demonstrations and schema.
- `completion`: JSON containing the accepted rationale followed by the reference answer.
- `source`: `"generate"` or `"rationalize"`, for auditing only.

Train on `prompt -> completion`, preserving this format and field order; do not
serialize the entire row into the model input. The reference answer replaces an
accepted equivalent answer in the completion. Hints are removed by rendering the
ordinary generation prompt again, rather than editing the hinted string. This
does not establish that a model's rationale is faithful or free of hint references.

The caller owns tokenization, loss masking, optimizer, step schedule, checkpoint
storage, provider settings, and transport retries. Use greedy or near-greedy
decoding to follow the paper; the official inference code uses temperature 0.01.
The example schedule can be replaced by your callback without changing the loop.

`Result.provider` is the **last trained model**, with `history` containing successful
round numbers and counts of direct/rationalized training examples. `stop_reason`
is `"budget"` or `"no_training_examples"`. An empty accepted set skips fine-tuning;
zero iterations returns the base provider. Held-out evaluation, plateau detection,
and best-checkpoint selection belong to the caller; training acceptance counts
are not held-out accuracy. For inference, use:

```python
solution = await agent.generate(new_input, provider=result.provider)
```

Here `agent` is the configured `STaR` instance used for training, retaining the
same task and demonstrations. `solution` contains `rationale` and `answer`.

## Official source mapping and adaptations

Sources inspected on 2026-09-14:

| Official source | Used in this implementation |
| --- | --- |
| [`iteration_train.py`](https://github.com/ezelikman/STaR/blob/main/iteration_train.py), `gen_train`, `train_model`, outer loop | Generate with the latest checkpoint; always fine-tune from `base_model_location`; replace the training set |
| [`device_inference.py`](https://github.com/ezelikman/STaR/blob/main/device_inference.py), `eval_examples`, `eval_output` | Rationalize failed inputs and accept only correct final answers |
| Same file, `examples_to_batch`, `question_to_context` | Separate hint-conditioned generation from hintless saved prompts and demonstrations |
| [`create_finetune_tfrecords.py`](https://github.com/ezelikman/STaR/blob/main/create_finetune_tfrecords.py) | Identifies the training-data/backend boundary, represented here by explicit prompt/completion rows |

The authors' implementation extends
[Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax), the
training implementation cited by the paper. This port adapts its algorithmic
decisions without importing its GPT-J/TPU, dataset, TFRecord, and cloud-storage
infrastructure. It does not install or execute those upstream training scripts.

Intentional adaptations: generic external Jinja prompts and a structured JSON
`Solution` replace benchmark-specific delimiters and answer extractors. Both
generated fields must be nonblank. There is one sample per input per phase,
all failures are rationalized by default, and all inputs are processed, including
sets smaller than upstream's batch size. All direct generations precede the
rationalization phase. There is no dataset subsampling, optional hint retention,
or automatic transport/parse recovery. This is an algorithm port, not an exact
reproduction of the paper's prompts or numerical experiments.

## Failures and checks

Generation, parsing, blank outputs, tool requests, evaluator failures, and trainer
failures propagate. No failure triggers hidden generations. `agent.calls` records
operation, iteration, input, rendered prompt, raw response when received,
correctness when checked, and generation/evaluation errors. Raw responses are
saved before parsing. `agent.training` retains the current completed training
batch, including when the trainer fails; `agent.history` records successful
training rounds. On a failure while collecting a batch, inspect `calls` for
partial work. These are in-memory records; persistence belongs to the caller.

Runs reset state to the original provider. Use one run at a time per instance.
Imports do not change Slick's process-global template root. Concurrent algorithms
requiring different template roots need separate processes; no shared Session
or conversation history is used.

```sh
optimizer/.venv/bin/python -B -m unittest tests.test_star
../slick/.venv/bin/ruff check star tests/test_star.py
../slick/.venv/bin/ruff format --check star tests/test_star.py
```

Tests use the shared scripted provider to verify the loop, model reset, filtering,
hint removal, dataset replacement, custom correctness, ablation, failure records,
and template rendering from another directory. These checks do not train a model
or demonstrate improved accuracy; no paper benchmark results have been reproduced.
