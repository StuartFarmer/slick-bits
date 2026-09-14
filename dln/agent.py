"""Train two stochastic language layers with sampled latent posteriors and prompt coordinate updates."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Example:
    input: str
    target: str


class DLN:
    """Own a DLN-2 forward pass, posterior weighting and prompt-level variational updates.

    evaluate(prediction, target) returns a finite nonnegative task loss.
    log_probability(rendered_context, target) returns the model's finite log
    likelihood of that target. It must use the same forward templates/model.
    Neither callback samples hidden states or chooses prompts.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[str, str], Awaitable[float]],
        log_probability: Callable[[str, str], Awaitable[float]],
    ):
        self.task, self.provider = task, provider
        self.evaluate, self.log_probability = evaluate, log_probability

    @prompt(template="hidden.j2")
    async def hidden(self, instruction: str, input: str, *, generated: str) -> str:
        """Execute the first language layer to obtain its latent text."""
        return generated

    @prompt(template="output.j2")
    async def output(self, instruction: str, input: str, hidden: str, *, generated: str) -> str:
        """Execute the output layer with both the original input and latent text."""
        return generated

    @prompt(template="posterior.j2")
    async def posterior(
        self,
        input: str,
        target: str,
        hidden: str,
        hidden_instruction: str,
        output_instruction: str,
        *,
        generated: str,
    ) -> str:
        """Sample a corrected latent explanation conditioned on the target output."""
        if not generated.strip():
            raise ValueError(f"empty posterior latent; response={generated!r}")
        return generated.strip()

    @prompt(template="propose_output.j2")
    async def propose_output(
        self, instruction: str, examples: list[dict], *, generated: str
    ) -> str:
        """Propose an output-layer instruction from forward execution errors."""
        if not generated.strip():
            raise ValueError(f"empty output instruction; response={generated!r}")
        return generated.strip()

    @prompt(template="propose_hidden.j2")
    async def propose_hidden(
        self, instruction: str, examples: list[dict], *, generated: str
    ) -> str:
        """Propose a first-layer instruction using the best posterior latent targets."""
        if not generated.strip():
            raise ValueError(f"empty hidden instruction; response={generated!r}")
        return generated.strip()

    async def _forward(self, examples):
        traces = []
        for example in examples:
            self.forward_calls += 1
            hidden = await self.hidden(
                self.hidden_instruction, example.input, provider=self.provider
            )
            self.forward_calls += 1
            output = await self.output(
                self.output_instruction, example.input, hidden, provider=self.provider
            )
            self.evaluations += 1
            loss = float(await self.evaluate(output, example.target))
            if not np.isfinite(loss) or loss < 0:
                raise ValueError("task loss must be finite and nonnegative")
            traces.append(
                {
                    "input": example.input,
                    "target": example.target,
                    "hidden": hidden,
                    "output": output,
                    "loss": loss,
                }
            )
        return traces

    async def _logp(self, layer, instruction, input, target, hidden=""):
        if layer == 1:
            context = await DLN.hidden.render(self, instruction, input)
        else:
            context = await DLN.output.render(self, instruction, input, hidden)
        self.likelihood_calls += 1
        value = float(await self.log_probability(context, target))
        if not np.isfinite(value):
            raise ValueError("model log likelihood must be finite")
        return value

    async def _infer_latents(
        self, traces, samples, temperature, include_prior, rewrite_errors_only, argmax_hidden
    ):
        latents, logits = [], []
        for trace in traces:
            row, weights = [], []
            for _ in range(samples):
                if rewrite_errors_only and trace["loss"] == 0:
                    latent = trace["hidden"]
                else:
                    self.optimizer_calls += 1
                    latent = await self.posterior(
                        trace["input"],
                        trace["target"],
                        trace["hidden"],
                        self.hidden_instruction,
                        self.output_instruction,
                        provider=self.provider,
                    )
                row.append(latent)
                value = await self._logp(
                    2, self.output_instruction, trace["input"], trace["target"], latent
                )
                if include_prior:
                    value += await self._logp(1, self.hidden_instruction, trace["input"], latent)
                weights.append(value)
            latents.append(row)
            logits.append(weights)
        weights = softmax(np.array(logits) / temperature, axis=1)
        best = [row[int(np.argmax(weight))] for row, weight in zip(latents, weights)]
        if argmax_hidden:
            latents, weights = [[latent] for latent in best], np.ones((len(best), 1))
        return latents, weights, best

    async def _update_output(self, traces, latents, weights, proposals):
        candidates = [self.output_instruction]
        for _ in range(proposals - 1):
            self.optimizer_calls += 1
            candidates.append(
                await self.propose_output(self.output_instruction, traces, provider=self.provider)
            )
        values = []
        for candidate in candidates:
            score = 0.0
            for trace, row, row_weights in zip(traces, latents, weights):
                for latent, weight in zip(row, row_weights):
                    score += weight * await self._logp(
                        2, candidate, trace["input"], trace["target"], latent
                    )
            values.append(float(score / len(traces)))
        self.output_instruction = candidates[int(np.argmax(values))]
        return {"candidates": candidates, "objectives": values}

    async def _update_hidden(
        self, traces, latents, weights, best_latents, proposals, wrong_penalty
    ):
        backward = [
            dict(trace, target_hidden=latent) for trace, latent in zip(traces, best_latents)
        ]
        candidates = [self.hidden_instruction]
        for _ in range(proposals - 1):
            self.optimizer_calls += 1
            candidates.append(
                await self.propose_hidden(self.hidden_instruction, backward, provider=self.provider)
            )
        values = []
        for candidate in candidates:
            score = 0.0
            for trace, row, row_weights in zip(traces, latents, weights):
                for latent, weight in zip(row, row_weights):
                    score += weight * await self._logp(1, candidate, trace["input"], latent)
                if wrong_penalty and trace["loss"] > 0:
                    score -= wrong_penalty * await self._logp(
                        1, candidate, trace["input"], trace["hidden"]
                    )
            values.append(float(score / len(traces)))
        self.hidden_instruction = candidates[int(np.argmax(values))]
        return {"candidates": candidates, "objectives": values}

    async def run(
        self,
        hidden_instruction: str,
        output_instruction: str,
        examples: Sequence[Example],
        *,
        iterations: int = 5,
        hidden_samples: int = 3,
        prompt_samples: int = 5,
        posterior_temperature: float = 1,
        include_prior: bool = True,
        rewrite_errors_only: bool = False,
        argmax_hidden: bool = False,
        wrong_hidden_penalty: float = 0,
    ) -> dict:
        """Execute both layers, infer posterior latents, update output then hidden prompts."""
        self.hidden_instruction, self.output_instruction = hidden_instruction, output_instruction
        self.optimizer_calls = self.forward_calls = self.evaluations = self.likelihood_calls = 0
        history = []
        for _ in range(iterations):
            traces = await self._forward(examples)
            if not any(trace["loss"] > 0 for trace in traces):
                history.append({"traces": traces, "converged": True})
                break
            latents, weights, best = await self._infer_latents(
                traces,
                hidden_samples,
                posterior_temperature,
                include_prior,
                rewrite_errors_only,
                argmax_hidden,
            )
            output_update = await self._update_output(traces, latents, weights, prompt_samples)
            hidden_update = await self._update_hidden(
                traces, latents, weights, best, prompt_samples, wrong_hidden_penalty
            )
            history.append(
                {
                    "traces": traces,
                    "latents": latents,
                    "weights": weights,
                    "output_update": output_update,
                    "hidden_update": hidden_update,
                }
            )
        return {
            "hidden_instruction": self.hidden_instruction,
            "output_instruction": self.output_instruction,
            "history": history,
            "optimizer_calls": self.optimizer_calls,
            "forward_calls": self.forward_calls,
            "evaluations": self.evaluations,
            "likelihood_calls": self.likelihood_calls,
        }
