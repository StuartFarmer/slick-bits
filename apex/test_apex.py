"""Offline checks: python test_apex.py (no credentials or model downloads)."""

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
from slick import prompts
from slick.providers import Provider

from apex import Config, Document, Embeddings, LinUCB, mutate, optimize, retrieve
from run import answer, check_split, evaluate, load_examples, predict


class Scripted(Provider):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.contexts = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.contexts.append(context)
        return next(self.replies), []


async def check():
    prompts.TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompts"
    assert await predict.render("Prefix\n", "Question", "\nSuffix") == "Prefix\nQuestion\nSuffix"
    for guided in (False, True):
        for examples in ([], [("Before.", "After."), ("Other.", "Better.")]):
            rendered = await mutate.render("Keep 3 labels.", examples, guided)
            assert "<original>Keep 3 labels.</original>" in rendered
            assert ("<original>Before.</original>" in rendered) == bool(guided and examples)
            assert ("professional sentence rephraser" in rendered) == guided

    original = "  Explain carefully.\r\nQ: Is 3.14 positive? Yes it is.\r\n(A) Yes\r\nA: Think.\r\n"
    doc = Document.split(original)
    assert doc.text == original
    assert all(doc.parts[i].strip() not in {"Q:", "A:", "(A)"} for i in doc.mutable)
    i = next(i for i in doc.mutable if doc.parts[i] == "Think.")
    revised = doc.replace(i, "Consider it.")
    assert revised.text == original.replace("Think.", "Consider it.")
    assert (
        Document.from_json(
            [{"text": "Q: ", "mutable": False}, {"text": "Think.", "mutable": True}]
        ).text
        == "Q: Think."
    )

    # Independent primal reference for the dual ridge implementation.
    h = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    rewards = np.array([0.2, -0.1, 0.3])
    bandit = LinUCB(regularization=0.7, alpha=0.05)
    for x, reward in zip(h, rewards):
        bandit.update(x, reward)
    x = np.array([[0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    a = h.T @ h + 0.7 * np.eye(3)
    expected = x @ np.linalg.solve(a, h.T @ rewards)
    expected += 0.05 * np.sqrt(np.sum(x * np.linalg.solve(a, x.T).T, axis=1))
    np.testing.assert_allclose(bandit.values(x), expected, atol=1e-12)
    np.testing.assert_allclose(LinUCB(0.7, 0.05).values(x), 0.05 / np.sqrt(0.7))

    history = [
        dict(before="good", after="better", reward=0.1),
        dict(before="bad", after="worse", reward=-0.2),
        dict(before="zero", after="same", reward=0.0),
        dict(before="far", after="distant", reward=0.9),
    ]
    vectors = {"good": [1.0, 0.0], "bad": [0.99, 0.01], "zero": [1.0, 0.0], "far": [0.0, 1.0]}
    examples = retrieve(
        np.array([1.0, 0.0]), history, lambda texts: np.array([vectors[t] for t in texts]), 4, 0.5
    )
    assert examples == [("good", "better"), ("worse", "bad")]
    assert (
        retrieve(
            np.array([1.0, 0.0]),
            history,
            lambda texts: np.array([vectors[t] for t in texts]),
            0,
            0.5,
        )
        == []
    )

    calls = []

    def encode(texts):
        calls.append(texts)
        return np.tile([3.0, 4.0], (len(texts), 1))

    embeddings = Embeddings(encode)
    np.testing.assert_allclose(embeddings(["a", "a"]), [[0.6, 0.8], [0.6, 0.8]])
    embeddings(["a", "b"])
    assert calls == [["a"], ["b"]]
    for invalid in ([[0.0, 0.0]], [[float("nan"), 1.0]], [1.0, 2.0]):
        try:
            Embeddings(lambda texts: invalid)(["a"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding accepted")

    provider = Scripted(["Better.", "Worse.", "Best.", "Best."])
    scores = {"Initial.": 0.5, "Better.": 0.8, "Worse.": 0.2, "Best.": 1.0}
    evaluated = []

    async def score(text):
        evaluated.append(text)
        return scores[text]

    result = await optimize(
        Document.split("Initial."),
        score,
        provider,
        lambda texts: np.tile([1.0, 0.0], (len(texts), 1)),
        Config(iterations=4, seed=3),
    )
    assert result["best"]["prompt"] == "Best."
    assert len(evaluated) == result["evaluations"] == 4  # duplicate is cached
    assert len(result["history"]) == 4
    assert [p["score"] for p in result["beam"]] == [1.0, 0.8, 0.5, 0.2]
    for step in result["history"]:
        assert np.isclose(step["reward"], scores[step["after"]] - step["parent_score"])
    assert "<original>" in provider.contexts[1]
    assert "<rephrased>" in provider.contexts[1]
    json.dumps(result, allow_nan=False)

    # Beam=1 rejects a worse candidate; malformed mutations cannot destroy a prompt.
    provider = Scripted(["Worse.", "", "Q: Hijack.", "Two\nlines", "Best.</rephrased>"])
    result = await optimize(
        Document.split("Initial."),
        score,
        provider,
        lambda texts: np.tile([1.0, 0.0], (len(texts), 1)),
        Config(iterations=5, beam_size=1, guided_mutation=False, random_probability=1.0),
    )
    assert result["best"]["score"] == 0.5
    assert result["evaluations"] == 2
    assert all(s["status"] == "invalid" for s in result["history"][1:])

    for kwargs in (
        {"iterations": -1},
        {"beam_size": 0},
        {"regularization": 0},
        {"alpha": float("nan")},
        {"random_probability": 2},
    ):
        try:
            Config(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)

    assert answer("Reasoning has 3 steps. The answer is (B).", "choice") == "B"
    assert answer("The answer is B.", "choice") == "B"
    assert answer("Work: 10 + 20.\n#### 1,234.00", "number") == answer("1234", "number")
    assert answer("The answer is invalid.", "text") == "invalid"
    assert answer("] ) }", "dyck") == "])}"
    assert answer("no numeric answer", "number") is None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.json"
        path.write_text(json.dumps({"examples": [{"input": "train", "target": "(A)"}]}))
        train = load_examples(path)
        path = Path(directory) / "data.jsonl"
        path.write_text(json.dumps({"question": "test", "answer": "(B)"}) + "\n")
        test = load_examples(path)
        check_split(train, test, "choice")
        try:
            check_split(train, [{"input": " TRAIN ", "target": "(A)"}], "choice")
        except ValueError:
            pass
        else:
            raise AssertionError("overlapping split accepted")

    provider = Scripted(["(A)", "Improve.", "(A)", "(B)"])

    async def training(text):
        return (await evaluate(text, train, provider, "\nQ: ", "\nA:", "choice"))["score"]

    result = await optimize(
        Document.split("Initial."), training, provider, encode, Config(iterations=1)
    )
    assert all("test" not in context for context in provider.contexts)
    held_out = await evaluate(result["best"]["prompt"], test, provider, "\nQ: ", "\nA:", "choice")
    assert held_out["score"] == 1.0
    assert provider.contexts[-1] == "Initial.\nQ: test\nA:"
    print(
        "APEX checks passed: formatting, LinUCB, history, beam, rewards, budgets, "
        "embeddings, parsing, holdout isolation, Slick."
    )


if __name__ == "__main__":
    asyncio.run(check())
