# APEER

`APEER(task, provider, respond, evaluate).run(initial_positive, initial_negative,
examples, ...)` alternates per-example feedback, aggregated refinement and
alignment against top/bottom prompt pairs. The two generated prompts are both
evaluated. Positive/negative membership uses the **fixed initial-positive score**;
each iteration starts from the best positive history entry. Higher scores win.
The result includes both histories and separate response, optimizer and evaluation
call counts. Exceptions propagate without hidden retries.

Examples are `Example(input, reference)` with caller-serialized fields.
`respond(prompt, input)` receives only the input; references go to feedback.
`evaluate` uses a separate validation set. This replaces the paper's query/passage/ranking
format with your task. Set Slick's template root to this folder's `prompts` once
at application startup.

Reconstructed from [paper Algorithm 1 and equations 2–6](https://arxiv.org/html/2406.14449v1).
The paper's [official repository](https://github.com/jincan333/APEER) returned 404
through both GitHub and its API during inspection. It could not be used, so this
is explicitly a paper reconstruction with task-neutral metaprompts. Reranking
dataset construction, BM25, model infrastructure and benchmark reproduction are
outside this algorithm port.
