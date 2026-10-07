from typing import List, Union
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from snackstack.agents.menu_agent import menu_agent_node
from snackstack.agents.orchestrator import orchestrator_node
from snackstack.agents.order_agent import order_agent_node
from snackstack.agents.synthesizer import synthesizer_node
from snackstack.logger import logger
from snackstack.state import StackState


def route_orchestrator(state: StackState) -> Union[str, List[Send]]:
    """Conditional edge routing based on Orchestrator's decision."""
    routes = state.get("route", ["menu_agent"])
    logger.info(f"Routing orchestrator output to: {routes}")

    if "menu_agent" in routes and "order_agent" in routes:
        logger.info("Parallel dispatch to both Menu Agent and Order Agent using Send().")
        return [
            Send("menu_agent", state),
            Send("order_agent", state),
        ]
    elif "order_agent" in routes:
        return "order_agent"
    else:
        return "menu_agent"


def build_graph():
    """Builds and compiles the SnackStack LangGraph workflow."""
    builder = StateGraph(StackState)

    # Add nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("menu_agent", menu_agent_node)
    builder.add_node("order_agent", order_agent_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Add edges
    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "menu_agent": "menu_agent",
            "order_agent": "order_agent",
        },
    )

    builder.add_edge("menu_agent", "synthesizer")
    builder.add_edge("order_agent", "synthesizer")
    builder.add_edge("synthesizer", END)

    # Checkpointer for conversation state & HITL interrupts
    memory = MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)
    return compiled_graph


graph = build_graph()
