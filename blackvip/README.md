# BlackVIP

Implements the coordinator/SPSA-GC loop from [BlackVIP](https://arxiv.org/abs/2303.14773),
grounded in official [trainers/blackvip.py](https://github.com/changdaeoh/BlackVIP/blob/main/trainers/blackvip.py).
The source uses input-dependent coordinator prompts, two-sided finite differences
with segmented uniform perturbations `[-1,-0.5] ∪ [0.5,1]`, optional gradient
averaging, power-law gain/radius schedules, and correction `g + beta * m` after
`m = beta * m + g`. These operations and multiclass cross entropy run locally.

Construct `BlackVIP(task, coordinate, forward, clip=...)` and call
`run(initial_parameters, batches, ...)`. Both model callbacks are async:
`coordinate(parameters, images)` is the frozen encoder plus trainable decoder's
forward calculation and returns an image-shaped prompt; `forward(prompted_images)`
returns class logits from the frozen target model. No gradient or optimizer callback
is needed. Each perturbation pair uses the same minibatch and original parameter
vector. Supply the source encoder/decoder and flatten only trainable coordinator
parameters to match its architecture; the default API supports other coordinators.

Images are augmented by `images + prompt_scale * prompt` before clipping. Default
clipping assumes unnormalized pixels in [0,1]; supply the source's normalized CLIP
clipping for normalized tensors. This is the explicit architecture/input adaptation.
The returned parameters are the last update, with gradient/momentum/schedule history.
There is no fabricated final score: `model_calls = 2 * averages * number_of_steps`.
Nonfinite observed cross entropy raises. Dependencies: NumPy and SciPy; actual
vision-model execution belongs to callbacks. No textual generation or templates
are required. Tests compare numerical gradients with an analytic logistic loss
and check correction, input dependence, and exact query count.
