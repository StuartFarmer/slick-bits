"""Fit PROMST's five Longformer regressors on independent 4:1 splits."""

import asyncio
import random
from collections.abc import Sequence

from .agent import Candidate, Heuristic


class LongformerFit:
    """Async fit callback; optional torch/transformers imports happen only at training.

    Each fit starts from the checkpoint on the accumulated measured archive.
    Refitting avoids leaking previous training labels into a new held-out split.
    Training and prediction run in a worker thread; use one fit at a time.
    Models remain on `device`. Epochs, learning rate, and batch size are local
    choices because the paper and released code do not specify them.
    """

    def __init__(
        self,
        checkpoint: str = "allenai/longformer-base-4096",
        *,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        max_length: int = 4096,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.checkpoint = checkpoint
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.max_length = learning_rate, max_length
        self.device, self.seed = device, seed
        self.models, self.splits = [], []
        self.tokenizer = None
        self.round = 0

    async def __call__(self, history: Sequence[Candidate]) -> Heuristic | None:
        # No meaningful 4:1 split yet: keep evaluating candidates without screening.
        if len(history) < 5:
            return None
        return await asyncio.to_thread(self._fit, tuple(history))

    def _batch(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)

    def _fit(self, history):
        import torch
        from transformers import AutoTokenizer, LongformerForSequenceClassification

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.models = []
        models, errors, splits = [], [], []
        device = torch.device(self.device)
        with torch.random.fork_rng():
            for member in range(5):
                seed = self.seed + 5 * self.round + member
                rng = random.Random(seed)
                indices = list(range(len(history)))
                rng.shuffle(indices)
                test_size = (len(history) + 4) // 5
                test, train = indices[:test_size], indices[test_size:]
                splits.append((tuple(train), tuple(test)))
                torch.manual_seed(seed)
                model = LongformerForSequenceClassification.from_pretrained(
                    self.checkpoint, num_labels=1, problem_type="regression"
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
                model.train()
                for _ in range(self.epochs):
                    rng.shuffle(train)
                    for offset in range(0, len(train), self.batch_size):
                        batch_indices = train[offset : offset + self.batch_size]
                        batch = self._batch([history[i].prompt for i in batch_indices])
                        labels = torch.tensor(
                            [history[i].score for i in batch_indices],
                            dtype=torch.float32,
                            device=device,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss = model(**batch, labels=labels).loss
                        if not torch.isfinite(loss).item():
                            raise ValueError("Longformer training loss must be finite")
                        loss.backward()
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del optimizer
                model.eval()
                absolute_errors = []
                with torch.no_grad():
                    for offset in range(0, len(test), self.batch_size):
                        batch_indices = test[offset : offset + self.batch_size]
                        batch = self._batch([history[i].prompt for i in batch_indices])
                        predicted = model(**batch).logits.flatten().tolist()
                        absolute_errors.extend(
                            abs(score - history[i].score)
                            for score, i in zip(predicted, batch_indices)
                        )
                models.append(model)
                errors.append(sum(absolute_errors) / len(absolute_errors))
        self.models, self.splits = models, splits
        self.round += 1

        async def predict(prompt: str) -> Sequence[float]:
            return await asyncio.to_thread(self._predict, prompt, models)

        return Heuristic(predict, tuple(errors))

    def _predict(self, prompt, models):
        import torch

        batch = self._batch([prompt])
        with torch.no_grad():
            return [model(**batch).logits.item() for model in models]
