"""One evaluation per process. Run model output ONLY inside the Docker image."""

import contextlib
import json
import os
import random
import sys

import numpy as np

from qube.problems import evaluate


def main():
    request = json.load(sys.stdin)
    random.seed(0)
    np.random.seed(0)
    scope = {"np": np, "__name__": "__candidate__"}
    # stdout is a result protocol; candidate diagnostics cannot fill host memory.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        exec(compile(request["code"], "<candidate>", "exec"), scope)
        name = "update_dist" if request["task"] == "tsp" else "priority"
        result = evaluate(request["task"], scope[name], request["instances"], request["iterations"])
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
