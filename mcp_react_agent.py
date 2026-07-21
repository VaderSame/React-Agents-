"""
Same ReAct loop, but driven by ChatOpenAI (pointed at OpenRouter).
The MCP half is untouched -- weather_server.py stays exactly as it was.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

QUESTION = "What's the weather in Toronto and Vancouver? Which one needs a heavier coat?"


def mcp_to_openai(tools):
    """MCP tool schema -> OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]

# StdioServerParameters is a configuration object used to launch and connect to a single local MCP server as a subprocess,
# whereas MultiServerMCPClient is a higher-level LangChain client designed to
# manage, route, and aggregate tools from multiple MCP servers simultaneously, supporting various network transports.


async def main():
    server = StdioServerParameters(command="python", args=["weather.py"])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print("Discovered tools:", [t.name for t in tools], "\n")

            llm = ChatOpenAI(
                model=os.getenv("MODEL_ID"),
                base_url=os.getenv("OPENROUTER_SERVER"),
                temperature=0,
            )
            # Attach the schemas. Returns a NEW runnable -- llm itself is unchanged.
            llm_with_tools = llm.bind_tools(mcp_to_openai(tools))

            messages = [HumanMessage(content=QUESTION)]

            while True:
                ai: AIMessage = await llm_with_tools.ainvoke(messages)
                messages.append(ai)

                if ai.content:
                    print("[thought/answer]", ai.content)

                # LangChain normalizes tool calls onto .tool_calls, already
                # JSON-parsed. Empty list == model is done.
                if not ai.tool_calls:
                    break

                for call in ai.tool_calls:
                    print(f"[action] {call['name']}({call['args']})")

                    result = await session.call_tool(call["name"], call["args"])
                    observation = "".join(
                        c.text for c in result.content if c.type == "text"
                    )
                    print(f"[observation] {observation}")

                    messages.append(
                        ToolMessage(
                            content=observation,
                            tool_call_id=call["id"],  # must echo it back
                        )
                    )


if __name__ == "__main__":
    asyncio.run(main())