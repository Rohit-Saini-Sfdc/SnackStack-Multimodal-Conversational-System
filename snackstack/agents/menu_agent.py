from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from snackstack.agents.prompts import MENU_AGENT_PROMPT
from snackstack.config import llm
from snackstack.logger import logger
from snackstack.state import StackState
from snackstack.tools.menu_tools import search_menu_catalog

menu_llm = llm.bind_tools([search_menu_catalog])


def menu_agent_node(state: StackState) -> dict:
    """Menu Agent node running an internal tool-calling loop to answer menu queries."""
    logger.info("Executing Menu Agent node...")

    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            user_query = str(last_msg.content)

    conversation = [
        SystemMessage(content=MENU_AGENT_PROMPT),
        HumanMessage(content=user_query),
    ]

    max_iterations = 5
    for iteration in range(max_iterations):
        response: AIMessage = menu_llm.invoke(conversation)
        conversation.append(response)

        if not response.tool_calls:
            logger.info(f"Menu Agent finished response on iteration {iteration + 1}")
            return {"menu_response": response.content}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            logger.info(f"Menu Agent calling tool '{tool_name}' with args: {tool_args}")
            if tool_name == "search_menu_catalog":
                query_arg = tool_args.get("query", user_query)
                tool_output = search_menu_catalog.invoke(query_arg)
            else:
                tool_output = f"Unknown tool: {tool_name}"

            conversation.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call_id)
            )

    # Fallback if max iterations reached
    logger.warning("Menu Agent reached max tool iterations cap.")
    return {
        "menu_response": (
            "Here is what I found on our menu catalog: "
            + search_menu_catalog.invoke(user_query)
        )
    }
