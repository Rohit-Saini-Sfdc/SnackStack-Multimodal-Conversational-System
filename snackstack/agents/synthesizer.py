from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from snackstack.agents.prompts import SYNTHESIZER_PROMPT
from snackstack.config import llm
from snackstack.logger import logger
from snackstack.state import StackState


def synthesizer_node(state: StackState) -> dict:
    """Synthesizer node that merges agent responses into a final user-facing reply."""
    logger.info("Executing Synthesizer node...")

    menu_resp = state.get("menu_response")
    order_resp = state.get("order_response")

    if menu_resp and order_resp:
        logger.info("Synthesizing both Menu Agent and Order Agent responses...")
        prompt = (
            f"User Query: {state.get('user_query', '')}\n\n"
            f"Menu Agent Response:\n{menu_resp}\n\n"
            f"Order Agent Response:\n{order_resp}"
        )
        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        final_text = str(response.content)
    elif menu_resp:
        logger.info("Using Menu Agent response as final answer.")
        final_text = menu_resp
    elif order_resp:
        logger.info("Using Order Agent response as final answer.")
        final_text = order_resp
    else:
        logger.warning("Neither Menu Agent nor Order Agent returned a response.")
        final_text = "I'm sorry, I couldn't process your request. How can I help you today?"

    return {
        "final_answer": final_text,
        "messages": [AIMessage(content=final_text)],
    }
