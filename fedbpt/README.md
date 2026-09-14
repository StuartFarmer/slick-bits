# FedBPT

Implements [FedBPT](https://arxiv.org/abs/2310.01467) using the official NVIDIA
[NVFlare research/fed-bpt](https://github.com/NVIDIA/NVFlare/tree/main/research/fed-bpt)
release, especially `src/fedbpt_train.py`, `src/global_es.py`, and
`src/LMForwardAPI.py`. Clients initialize local CMA distributions from the server
mean/covariance/scale, perform local search, and send final means, measured losses,
and their per-step sigmas. The server ranks these means with its own CMA update.
Its effective step scale is `sqrt(mean_clients(sum(local_sigmas**2)/local_mu))`;
the resulting multiplicative sigma change is transferred to the next local scale.
This is not FedAvg of prompt vectors or covariance matrices.

`FedBPT(task, evaluate, clients, initial_prompt, projection).run(...)` accepts one
shared fixed projection and async `evaluate(client_id, prompt_array, perturbed)`.
The evaluator returns lower-is-better original or perturbed-input loss. With
`regularize=True`, the local algorithm computes their ratio itself, matching the
source's anti-overfitting objective. The caller creates and fixes each client's
perturbed dataset using the paper's input-token masking/replacement. With
regularization disabled, perturbed losses are never queried. Sampling, all local
and server CMA updates, ratios and sigma aggregation remain inside the agent.

Source-related choices: client distributions copy mean/covariance/scale with fresh
paths; the server retains all participating clients as positive-weight parents.
The `bbt/cma.py` core implements standard positive-weight full-covariance CMA, not
pycma's active negative weights/bounds/restarts. A single seeded NumPy RNG replaces
the release's process-specific random streams. The release's ineffective sigma
floor (immediately undone by its following comparison) is omitted. Distributed
transport, checkpointing and test-data evaluation are outside the mathematical
port; it does not claim privacy guarantees or reproduce a federated deployment.

Each round makes `clients_per_round * (population * local_steps + 1)` fitness
queries, doubled with perturbation regularization. Reports expose the client
fitnesses, sigmas, aggregate scale, adaptation ratio and final server mean. The
returned prompt is projected from that mean and has no asserted global fitness.
Nonfinite observed fitness raises. Dependencies: NumPy plus sibling `bbt`; no LLM
generation/provider/templates. Tests verify ratio call counts, server variance
calculation, client sampling and the no-perturbation ablation.
