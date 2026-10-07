from typing import Annotated, List, Optional, TypedDict
from langgraph.graph.message import add_messages


class StackState(TypedDict):
    """Shared state dictionary passed through the LangGraph agents."""

    messages: Annotated[list, add_messages]
    user_query: str
    route: List[str]
    menu_response: Optional[str]
    order_response: Optional[str]
    final_answer: Optional[str]
