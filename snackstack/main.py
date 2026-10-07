import argparse
import sys
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from snackstack.graph import graph
from snackstack.logger import logger
from snackstack.voice.recorder import record_and_transcribe
from snackstack.voice.speaker import speak_text


class SnackStackAssistant:
    """Wrapper class around compiled LangGraph workflow managing conversation threads and HITL interrupts."""

    def __init__(self):
        self.graph = graph
        self.thread_id = str(uuid.uuid4())

    def reset_conversation(self):
        """Resets the conversation thread ID to start a fresh interaction."""
        self.thread_id = str(uuid.uuid4())
        logger.info(f"Conversation reset. New thread_id: {self.thread_id}")

    def ask(self, user_input: str) -> str:
        """Processes user input, handles graph invocation, resumes interrupts if pending, and returns final answer.

        Args:
            user_input: Natural language query or response to interrupt.

        Returns:
            Assistant response string or interrupt question.
        """
        config = {"configurable": {"thread_id": self.thread_id}}
        current_state = self.graph.get_state(config)

        # Check if the graph is currently paused on an interrupt
        if current_state.next:
            logger.info("Resuming execution from pending interrupt...")
            result = self.graph.invoke(Command(resume=user_input), config)
        else:
            logger.info(f"Invoking graph with user query: '{user_input}'")
            initial_state = {
                "user_query": user_input,
                "messages": [HumanMessage(content=user_input)],
            }
            result = self.graph.invoke(initial_state, config)

        # Re-check state after invocation for any new pending interrupts
        new_state = self.graph.get_state(config)
        if new_state.next:
            for task in new_state.tasks:
                if task.interrupts:
                    interrupt_question = task.interrupts[0].value
                    return f"❓ {interrupt_question}"

        # Extract final answer
        final_answer = result.get("final_answer")
        if not final_answer and "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                final_answer = str(last_msg.content)

        return final_answer or "No response generated."


def run_text_loop(voice_in: bool = False, voice_out: bool = False):
    """CLI Interactive REPL Loop."""
    assistant = SnackStackAssistant()

    print("\n" + "=" * 60)
    print(" 🍔 Welcome to SnackStack Food Delivery Assistant! 🍕 ")
    print("=" * 60)
    print("Commands:")
    print("  - Type your query (e.g., 'Show me vegan dishes' or 'Track ORD-201')")
    print("  - Type 'reset' to start a fresh conversation session.")
    print("  - Type 'quit' or 'exit' to exit.")
    if voice_in:
        print("  - Voice Input: ENABLED (Speech-to-Text via OpenAI Whisper)")
    if voice_out:
        print("  - Voice Output: ENABLED (Text-to-Speech via OpenAI TTS)")
    print("=" * 60 + "\n")

    while True:
        try:
            if voice_in:
                input("Press ENTER to start recording... ")
                user_input = record_and_transcribe()
                if not user_input:
                    print("⚠️ Could not transcribe audio. Please type your query:")
                    user_input = input("\nYou: ").strip()
                else:
                    print(f"\nYou (Voice): {user_input}")
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in ["quit", "exit"]:
                print("\nThank you for using SnackStack! Goodbye! 👋\n")
                break

            if lowered == "reset":
                assistant.reset_conversation()
                print("\nSession reset successfully. How can I help you today?\n")
                continue

            response = assistant.ask(user_input)
            print(f"\nSnackStack: {response}\n")

            if voice_out and response:
                speak_text(response)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting SnackStack. Goodbye! 👋\n")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="SnackStack Food Delivery Assistant")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable full voice mode (Voice Input STT + Voice Output TTS)",
    )
    parser.add_argument(
        "--voice-out",
        action="store_true",
        help="Enable text input with voice output TTS",
    )

    args = parser.parse_args()
    voice_in = args.voice
    voice_out = args.voice or args.voice_out

    run_text_loop(voice_in=voice_in, voice_out=voice_out)


if __name__ == "__main__":
    main()
