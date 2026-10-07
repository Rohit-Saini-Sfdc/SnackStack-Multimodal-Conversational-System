from typing import List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from snackstack.agents.prompts import ORCHESTRATOR_PROMPT
from snackstack.config import llm
from snackstack.logger import logger
from snackstack.state import StackState


class RouteDecision(BaseModel):
    routes: List[Literal["menu_agent", "order_agent"]] = Field(
        description="List of target agents to route the query to: 'menu_agent', 'order_agent', or both."
    )
    reasoning: str = Field(
        description="Explanation of why this routing decision was made."
    )


orchestrator_llm = llm.with_structured_output(RouteDecision)


def orchestrator_node(state: StackState) -> dict:
    """Orchestrator node that classifies user query and decides target agent routes."""
    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            user_query = str(last_msg.content)

    logger.info(f"Orchestrator evaluating query: '{user_query}'")

    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=user_query),
    ]

    try:
        decision: RouteDecision = orchestrator_llm.invoke(messages)
        routes = decision.routes
        logger.info(f"Orchestrator decision: routes={routes}, reasoning='{decision.reasoning}'")
    except Exception as e:
        logger.error(f"Orchestrator structured output error: {e}. Defaulting to menu_agent.")
        routes = ["menu_agent"]

    if not routes:
        routes = ["menu_agent"]

    return {"route": routes}
