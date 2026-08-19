"""SLM inference toolbox, ported to the DSS LLM Mesh.

One module per technique from the repo's `inference_methods/`, sharing a single
agentic episode runner over `hscode_env.HSCodeEnv`. Each method exposes
`run(description, **kwargs) -> dict` with a uniform result shape, so a DSS agent
is a three-line wrapper and the flow can score every method the same way.
"""

from .methods import METHODS, run_method

__all__ = ["METHODS", "run_method"]
