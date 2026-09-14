# BPO

`BPO(task, provider, evaluate).run(samples=1)` runs the learned prompt rewriter
and asynchronously scores the resulting instruction. Supply a provider serving
the authors' **THUDM/BPO** checkpoint (and returning only the completion). A generic
chat model with the same template is a different experiment. The source uses
top-p 0.9, temperature 0.6, and a 1024-token completion limit for stable inference;
configure those on your provider. The original Llama instruction wrapper is kept.

Sources: [paper](https://arxiv.org/abs/2311.04155), official
[inference code](https://github.com/thu-coai/bpo/blob/9bf541587d1456fda6ffa46001ab545e892f4a2c/src/infer_example.py).
This ports **deployment inference**, not preference-data construction or checkpoint
training. Multiple samples with external best-score selection extend the original
single-sample stable path; upstream's aggressive token-blocking mode is not ported.
Scores are finite, higher is better, stable ties select the first sample. Provider,
parsing and evaluator errors propagate without retries; agent counters record attempts.

Set `slick.prompts.TEMPLATE_ROOT` once to this folder's absolute `prompts/` path
before running. Tests use scripted completions; checkpoint quality is not measured.
