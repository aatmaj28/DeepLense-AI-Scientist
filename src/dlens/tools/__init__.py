# tools/__init__.py
from dlens.tools._rest_client import (
    RestClient,
    RestClientConfig,
    make_api_request,
    get_api_key,
)
from dlens.tools._base import (
    AbstractToolInputSchema,
    AbstractToolOutputSchema,
    AbstractBaseTool,
)
from dlens.tools._mcp_wrapper import mcp_rest_tool
from dlens.tools._sim_backends import (
    DeepLensBackend,
    MockBackend,
    SimBackend,
    deeplense_available,
    get_backend,
)
from dlens.tools._sim_runner import execute_simulation
from dlens.tools._simulation import SimDeps, register_simulation_tool
from dlens.tools._model_design import recommend_baseline, register_model_design_tool

__all__ = [
    "RestClient",
    "RestClientConfig",
    "make_api_request",
    "get_api_key",
    "AbstractToolInputSchema",
    "AbstractToolOutputSchema",
    "AbstractBaseTool",
    "mcp_rest_tool",
    "SimBackend",
    "MockBackend",
    "DeepLensBackend",
    "deeplense_available",
    "get_backend",
    "execute_simulation",
    "SimDeps",
    "register_simulation_tool",
    "recommend_baseline",
    "register_model_design_tool",
]
