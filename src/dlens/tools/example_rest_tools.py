# Example REST tools demonstrating how to build MCP-compatible API wrappers.
# Uses NASA's public APIs as a reference implementation.
# credits to https://github.com/portkeys/nasa-mcp

from __future__ import annotations
from dlens.tools._rest_client import make_api_request, get_api_key
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def get_nasa_apod(date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve NASA's Astronomy Picture of the Day (APOD).

    Args:
        date: Optional date in YYYY-MM-DD format. If None, returns today's APOD.

    Returns:
        APOD JSON response dict if successful, else None.

    Example:
        apod = get_nasa_apod(date="2023-01-01")
        if apod:
            print(apod["title"])
    """
    base_url = "https://api.nasa.gov/planetary/apod"

    api_key = get_api_key("NASA_API_KEY", fallback_key="DEMO_KEY")

    params: Dict[str, Any] = {"api_key": api_key}
    if date:
        params["date"] = date

    return make_api_request(base_url, params, timeout=10)


def search_nasa_images(query: str, size: int = 3) -> Optional[Dict[str, Any]]:
    """
    Search NASA's Image and Video Library for images matching the query.

    Args:
        query: Search term for NASA's image library.
        size: Number of results to return (page_size). Defaults to 3.

    Returns:
        Search results JSON response dict if successful, else None.

    Example:
        results = search_nasa_images("Mars rover", size=5)
        if results:
            for item in results["collection"]["items"]:
                print(item["data"][0]["title"])
    """
    base_url = "https://images-api.nasa.gov/search"

    page_size = max(1, min(int(size), 100))

    params: Dict[str, Any] = {
        "q": query,
        "media_type": "image",
        "page": 1,
        "page_size": page_size,
    }

    return make_api_request(base_url, params, timeout=15)
