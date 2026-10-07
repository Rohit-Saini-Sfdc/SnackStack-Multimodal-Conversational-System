import re
from typing import Optional
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import interrupt
from snackstack.agents.prompts import ORDER_AGENT_PROMPT
from snackstack.config import llm
from snackstack.logger import logger
from snackstack.state import StackState
from snackstack.tools.order_tools import get_order_status

order_llm = llm.bind_tools([get_order_status])


def extract_identifier(text: str) -> Optional[str]:
    """Extracts Order ID, Tracking ID, or Email from text using regex patterns."""
    if not text:
        return None

    # Check Order ID (e.g., ORD-201)
    match_ord = re.search(r"\bORD-\d+\b", text, re.IGNORECASE)
    if match_ord:
        return match_ord.group(0).upper()

    # Check Tracking ID (e.g., SS201TRK)
    match_trk = re.search(r"\bSS\d+TRK\b", text, re.IGNORECASE)
    if match_trk:
        return match_trk.group(0).upper()

    # Check Email address
    match_email = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text, re.IGNORECASE)
    if match_email:
        return match_email.group(0).lower()

    return None


def order_agent_node(state: StackState) -> dict:
    """Order Agent node with HITL interrupt if no order identifier is present."""
    logger.info("Executing Order Agent node...")

    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            user_query = str(last_msg.content)

    identifier = extract_identifier(user_query)

    # If no identifier is found in user_query, pause execution using interrupt()
    if not identifier:
        logger.info("No order identifier found in query. Triggering HITL interrupt.")
        resumed_input = interrupt(
            "I couldn't find an Order ID or Tracking ID in your request. "
            "Please provide your Order ID (e.g. ORD-201), Tracking ID (e.g. SS201TRK), or Email address:"
        )
        logger.info(f"Order Agent resumed with user input: '{resumed_input}'")
        resumed_text = str(resumed_input)
        identifier = extract_identifier(resumed_text) or resumed_text.strip()
        user_query = f"{user_query} (Order details: {identifier})"

    logger.info(f"Order Agent proceeding with identifier: '{identifier}'")

    conversation = [
        SystemMessage(content=ORDER_AGENT_PROMPT),
        HumanMessage(content=f"Lookup order for identifier: {identifier}. Context query: {user_query}"),
    ]

    max_iterations = 5
    for iteration in range(max_iterations):
        response: AIMessage = order_llm.invoke(conversation)
        conversation.append(response)

        if not response.tool_calls:
            logger.info(f"Order Agent finished response on iteration {iteration + 1}")
            return {"order_response": response.content}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            logger.info(f"Order Agent calling tool '{tool_name}' with args: {tool_args}")
            if tool_name == "get_order_status":
                id_arg = tool_args.get("identifier", identifier)
                tool_output = get_order_status.invoke(id_arg)
            else:
                tool_output = f"Unknown tool: {tool_name}"

            conversation.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call_id)
            )

    # Fallback if max iterations reached
    logger.warning("Order Agent reached max tool iterations cap.")
    fallback_status = get_order_status.invoke(identifier)
    return {"order_response": f"Order Status for {identifier}:\n{fallback_status}"}
