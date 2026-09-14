"""Recover a prompt from desired documents using inverse-likelihood GCG."""

from collections.abc import Awaitable, Callable

import numpy as np
from scipy.special import logsumexp, softmax
from slick import prompt
from slick.providers import Provider


class PROPANE:
    """Forward/backward callbacks expose only model logits and their VJP.

    forward(one_hot_prompt, documents) returns [documents, tokens, vocabulary]
    logits predicting each document token. backward receives their cotangent and
    returns [prompt_tokens, vocabulary] derivatives. Model context stays fixed.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        forward: Callable[[np.ndarray, np.ndarray], Awaitable[np.ndarray]],
        backward: Callable[[np.ndarray, np.ndarray, np.ndarray], Awaitable[np.ndarray]],
        encode: Callable[[str], np.ndarray],
        decode: Callable[[np.ndarray], str],
        vocabulary_size: int,
    ):
        self.task, self.provider, self.forward, self.backward = task, provider, forward, backward
        self.encode, self.decode, self.vocabulary_size = encode, decode, vocabulary_size

    @prompt(template="warm_start.j2")
    async def warm_start(self, documents: list[str], *, generated: str) -> str:
        if not generated.strip():
            raise ValueError(f"empty generated prompt; response={generated!r}")
        return generated.strip()

    async def _objective(self, tokens, documents, gradient=False):
        weights = np.eye(self.vocabulary_size)[tokens]
        self.forward_calls += 1
        logits = np.asarray(await self.forward(weights, documents))
        log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
        targets = np.take_along_axis(log_probs, documents[..., None], axis=-1)[..., 0]
        loss = float(-targets.mean())
        if not np.isfinite(loss):
            raise ValueError("measured document likelihood must be finite")
        derivative = None
        if gradient:
            cotangent = softmax(logits, axis=-1)
            rows, positions = np.indices(documents.shape)
            cotangent[rows, positions, documents] -= 1
            cotangent /= documents.size
            self.backward_calls += 1
            derivative = np.asarray(await self.backward(weights, documents, cotangent))
            if not np.isfinite(derivative).all():
                raise ValueError("measured prompt gradient must be finite")
        return loss, targets, derivative

    async def run(
        self,
        documents: np.ndarray,
        initial_tokens: np.ndarray | None = None,
        *,
        iterations: int = 100,
        top_k: int = 256,
        allowed_tokens: np.ndarray | None = None,
        teacher_log_probs: np.ndarray | None = None,
        seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(seed)
        self.forward_calls = self.backward_calls = self.optimizer_calls = 0
        self.history = []
        if initial_tokens is None:
            self.optimizer_calls += 1
            text = await self.warm_start(
                [self.decode(row) for row in documents], provider=self.provider
            )
            initial_tokens = self.encode(text)
        current = np.array(initial_tokens, dtype=int, copy=True)
        vocabulary = np.arange(self.vocabulary_size) if allowed_tokens is None else allowed_tokens
        best_loss, best_targets, _ = await self._objective(current, documents)
        best = current.copy()
        for _ in range(iterations):
            _, _, gradient = await self._objective(current, documents, gradient=True)
            top = vocabulary[np.argsort(gradient[:, vocabulary], axis=1, kind="stable")[:, :top_k]]
            proposals = []
            for position in range(len(current)):
                proposal = current.copy()
                proposal[position] = rng.choice(top[position])
                loss, targets, _ = await self._objective(proposal, documents)
                proposals.append((loss, proposal, targets))
            loss, current, targets = min(proposals, key=lambda item: item[0])
            if loss < best_loss:
                best_loss, best, best_targets = loss, current.copy(), targets
            self.history.append(
                {
                    "tokens": current.copy(),
                    "loss": loss,
                    "proposals": [(item[1].copy(), item[0]) for item in proposals],
                }
            )
        kl = (
            None
            if teacher_log_probs is None
            else float(np.mean(np.sum(teacher_log_probs - best_targets, axis=1)))
        )
        return {
            "tokens": best,
            "prompt": self.decode(best),
            "loss": best_loss,
            "empirical_kl": kl,
            "current": current,
            "history": self.history,
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "optimizer_calls": self.optimizer_calls,
        }
