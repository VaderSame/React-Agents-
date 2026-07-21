"""
A minimal MCP server that exposes one tool: get_weather.

Run standalone to sanity-check:  python weather_server.py
(It will just sit there waiting on stdin -- that's correct, it speaks
JSON-RPC over stdio. The agent will launch it as a subprocess.)

pip install "mcp[cli]"
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

# Static "database". In real life this would be an API call.
FAKE_WEATHER = {
    "toronto": {"temp_c": -3, "condition": "light snow", "wind_kph": 22},
    "london": {"temp_c": -5, "condition": "overcast", "wind_kph": 15},
    "vancouver": {"temp_c": 9, "condition": "rain", "wind_kph": 8},
}


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city, e.g. "Toronto"
    """
    data = FAKE_WEATHER.get(city.strip().lower())
    if data is None:
        return f"No weather data available for {city}."
    return (
        f"{city}: {data['temp_c']}degC, {data['condition']}, "
        f"wind {data['wind_kph']} km/h"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")