"""Talking to a decision service, with nothing else attached.

The client is the whole of this package an environment needs, and an environment
runs where its own dependencies are: for ScienceWorld that is a container with a
JVM and no GPU. Keeping the client beside the service put `import torch` at the
top of the module an environment had to import, so the sandbox either carried
torch to send JSON or did not run. It lives here instead, and imports nothing
beyond the standard library.
"""

import json
from urllib import error, request


def _decode_hidden(encoded):
    """base64 float32 back to a list, without importing numpy.

    The client is deliberately standard-library only: an environment runs where
    its own dependencies are, and for ScienceWorld that was a container with a JVM
    and no GPU. A caller that wants an array can make one.
    """
    import array
    import base64

    raw = base64.b64decode(encoded["data"])
    values = array.array("f")
    values.frombytes(raw)
    return list(values)


class DecisionClient:
    """The service as seen from wherever the browser is."""

    def __init__(self, base_url, timeout=900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, route, body=None):
        url = f"{self.base_url}{route}"
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        try:
            with request.urlopen(request.Request(url, data=data, headers=headers),
                                 timeout=self.timeout) as response:
                return json.loads(response.read())
        except error.HTTPError as problem:
            raise RuntimeError(
                f"the decision service refused {route}: {problem.code} "
                f"{problem.read()[:300].decode('utf-8', 'replace')}") from problem
        except error.URLError as problem:
            raise RuntimeError(
                f"no decision service at {self.base_url}: {problem.reason}. Start one "
                f"with python -m inference.decision_service beside the SGLang server."
            ) from problem

    def health(self):
        return self._call("/health")

    def fast(self, state, actions, want_hidden=False):
        """The fast distribution, and the head's input state when asked for it.

        `want_hidden` is for training: CDPO learns from the readout hidden state
        at every step, and the engine cannot be in the browser's process --
        Playwright's sync API runs an event loop on the calling thread and
        SGLang's `Engine` drives its own. One vector is about 20 KB of base64,
        which is not the 48 MB this service exists to avoid: that was sglang's own
        server encoding every prompt position.
        """
        answer = self._call("/fast", {"state": state, "actions": list(actions),
                                      "want_hidden": bool(want_hidden)})
        if want_hidden and answer.get("hidden"):
            answer["hidden_state"] = _decode_hidden(answer["hidden"])
        return answer

    def slow(self, state, actions, budget):
        return self._call("/slow", {"state": state, "actions": list(actions),
                                    "budget": int(budget)})

    def solve(self, state, instructions, budget):
        """The slow mode writing its answer, with no candidate set.

        What `slow` returns is a distribution over the actions it was handed, so
        escalating a step whose right action is text bought a longer thought and
        no new capability. This is the other shape: the model writes the action
        and the caller parses it.
        """
        return self._call("/solve", {"state": state, "instructions": instructions,
                                     "budget": int(budget)})


def head_device(choice="auto"):
    """Where the decision head runs. The same device as the backbone by default.

    The head is part of the model: 5128 -> 256 -> 4, about 1.3M parameters, and
    it reads a hidden state the backbone has just produced. It ran on the CPU
    here only because `DecisionHead.load` defaults to one and nothing said
    otherwise.
    """
    if choice != "auto":
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001  no torch is a CPU deployment by definition
        return "cpu"
