"""Filter self-generated CoT demonstrations, retrieve neighbors and vote over choices."""

import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
from slick import prompt
from slick.providers import Provider


@dataclass(frozen=True)
class Question:
    text: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class Example:
    question: Question
    answer: int


@dataclass(frozen=True)
class Solution:
    question: Question
    response: str
    answer: int


class MedPrompt:
    """Use the original MMLU KNN path with random choice shuffles and majority vote.

    Training labels are used only to filter generated rationales. evaluate sees
    the original query and a zero-based answer index after permutation inversion.
    """

    def __init__(
        self,
        task: str,
        provider: Provider,
        evaluate: Callable[[Question, int], Awaitable[float]],
        embed: Callable[[Sequence[str]], Awaitable[np.ndarray]],
    ):
        self.task, self.provider, self.evaluate, self.embed = task, provider, evaluate, embed

    @prompt(template="self_cot.j2")
    async def self_cot(self, description: str, *, generated: str) -> str:
        return generated

    @prompt(template="answer.j2")
    async def answer(
        self, description: str, demonstrations: Sequence[dict], *, generated: str
    ) -> str:
        return generated

    @staticmethod
    def _description(question, order):
        return (
            question.text
            + "\n\n"
            + "\n".join(f"{chr(65 + i)}. {question.choices[j]}" for i, j in enumerate(order))
        )

    @staticmethod
    def _parse(response, size):
        tail = response.split("\nAnswer: ", 1)[-1]
        found = [i for i in range(size) if f"[{chr(65 + i)}]" in tail]
        return found[0] if len(found) == 1 else None

    async def _build_pool(self, training, attempts):
        self.pool = []
        for example in training:
            solutions = []
            for _ in range(attempts):
                self.generations += 1
                response = await self.self_cot(
                    self._description(example.question, range(len(example.question.choices))),
                    provider=self.provider,
                )
                predicted = self._parse(response, len(example.question.choices))
                if predicted == example.answer:
                    solutions.append(Solution(example.question, response, predicted))
            if solutions:
                self.pool.append(solutions)
        if not self.pool:
            raise ValueError("no correct generated demonstrations")
        self.embeddings = np.asarray(
            await self.embed([rows[0].question.text for rows in self.pool])
        )
        if not np.isfinite(self.embeddings).all():
            raise ValueError("embeddings must be finite")

    async def _neighbors(self, query, k):
        vector = np.asarray(await self.embed([query.text]))[0]
        if not np.isfinite(vector).all():
            raise ValueError("query embedding must be finite")
        denom = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vector)
        similarity = np.divide(
            self.embeddings @ vector, denom, out=np.zeros(len(self.pool)), where=denom > 0
        )
        eligible = [
            i
            for i in np.argsort(-similarity, kind="stable")
            if query.text not in self.pool[i][0].question.text
        ]
        return eligible[:k][::-1]

    async def predict(self, query: Question, *, neighbors: int = 5, ensemble: int = 5) -> dict:
        selected = await self._neighbors(query, neighbors)
        votes, trials = [], []
        for _ in range(ensemble):
            order = list(range(len(query.choices)))
            self.rng.shuffle(order)
            demonstrations = []
            for index in selected:
                solution = self.rng.choice(self.pool[index])
                demonstrations.append(
                    {
                        "question": self._description(
                            solution.question, range(len(solution.question.choices))
                        ),
                        "answer": solution.response,
                    }
                )
            self.generations += 1
            response = await self.answer(
                self._description(query, order), demonstrations, provider=self.provider
            )
            choice = self._parse(response, len(order))
            actual = None if choice is None else order[choice]
            if actual is not None:
                votes.append(actual)
            trials.append({"order": order, "response": response, "answer": actual})
        if not votes:
            raise ValueError("all ensemble answers were unparseable")
        answer = Counter(votes).most_common(1)[0][0]
        return {"answer": answer, "trials": trials, "neighbors": selected}

    async def run(
        self,
        training: Sequence[Example],
        queries: Sequence[Question],
        *,
        rationale_attempts: int = 1,
        neighbors: int = 5,
        ensemble: int = 5,
        seed: int = 0,
    ) -> dict:
        self.rng = random.Random(seed)
        self.generations = self.evaluations = 0
        await self._build_pool(training, rationale_attempts)
        predictions = []
        for query in queries:
            prediction = await self.predict(query, neighbors=neighbors, ensemble=ensemble)
            self.evaluations += 1
            score = float(await self.evaluate(query, prediction["answer"]))
            if not math.isfinite(score):
                raise ValueError("prediction score must be finite")
            predictions.append({**prediction, "score": score})
        return {
            "pool": self.pool,
            "predictions": predictions,
            "generations": self.generations,
            "evaluations": self.evaluations,
        }
