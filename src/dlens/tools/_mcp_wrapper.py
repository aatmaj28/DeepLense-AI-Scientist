"""
A generalizable REST -> FastMCP wrapper.

# Your REST functions are synchronous (blocking), but Pydantic AI expects async functions for MCP tools, so we need to wrap the original function in an async function and run it in a thread to avoid blocking the agent loop.

Adds:
- Sync → async bridging
- JSON-safe output handling
- Uniform error handling
- Signature copying

Assumptions:
- Your REST functions return Optional[Dict[str, Any]] (None on failure).
- You want MCP tools to return JSON strings (pretty) or an error dict.
"""

from __future__ import annotations

import json
import inspect
import asyncio
from typing import Any, Callable, Optional
from mcp.server.fastmcp import FastMCP
from loguru import logger


def _safe_json_dumps(obj: Any, *, indent: int = 2) -> str:
    """JSON dump with a couple of safe defaults for tool outputs."""
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def mcp_rest_tool(
    *,
    mcp: FastMCP,
    name: str,
    rest_fn: Callable[..., Any],
    doc: Optional[str] = None,
    indent: int = 2,
    none_error: str = "Request failed",
    wrap_exceptions: bool = True,
) -> Callable[..., Any]:
    tool_doc = doc or (rest_fn.__doc__ or "")

    # handler function to wrap the original function and return the result in a JSON string.
    # Async and error handling.
    async def _handler(*args: Any, **kwargs: Any) -> Any:
        try:
            if inspect.iscoroutinefunction(rest_fn):
                result = await rest_fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(rest_fn, *args, **kwargs)

            if result is None:
                return {"error": none_error}

            if isinstance(result, (dict, list)):
                return _safe_json_dumps(result, indent=indent)

            return _safe_json_dumps(result, indent=indent)

        except Exception as e:
            if not wrap_exceptions:
                raise
            logger.exception(
                "Tool {} failed calling {}",
                name,
                getattr(rest_fn, "__name__", str(rest_fn)),
            )
            return {"error": f"{name} failed: {e}"}

    _handler.__name__ = name
    _handler.__doc__ = tool_doc

    # Signature copying to make the generic wrapper look like the original function to the Pydantic AI model.
    try:
        sig = inspect.signature(rest_fn)
        # Resolve string annotations to actual types so Pydantic can build its model
        type_hints = inspect.get_annotations(rest_fn, eval_str=True)
        new_params = []
        for p in sig.parameters.values():
            if p.name in type_hints:
                new_params.append(p.replace(annotation=type_hints[p.name]))
            else:
                new_params.append(p)
        sig = sig.replace(parameters=new_params, return_annotation=Any)
        _handler.__signature__ = sig
    except Exception as e:
        logger.exception(f"Error resolving signature for tool {name}: {e}")
        pass

    return mcp.tool(name)(_handler)
