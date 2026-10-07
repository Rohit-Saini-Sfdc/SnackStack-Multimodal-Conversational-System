import sys
import unittest
from snackstack.main import SnackStackAssistant
from snackstack.tools.menu_tools import search_menu_catalog
from snackstack.tools.order_tools import get_order_status


class TestSnackStack(unittest.TestCase):

    def test_menu_tool_directly(self):
        result = search_menu_catalog.invoke("vegan")
        print("\n--- Direct Menu Tool Test (Vegan) ---")
        print(result)
        self.assertIn("Vegan", result)

    def test_order_tool_by_id(self):
        result = get_order_status.invoke("ORD-201")
        print("\n--- Direct Order Tool Test (ORD-201) ---")
        print(result)
        self.assertIn("Butter Chicken", result)
        self.assertIn("Priya Nair", result)

    def test_order_tool_by_email(self):
        result = get_order_status.invoke("priya@example.com")
        print("\n--- Direct Order Tool Test (Email) ---")
        print(result)
        self.assertIn("ORD-201", result)

    def test_assistant_menu_query(self):
        assistant = SnackStackAssistant()
        res = assistant.ask("Show me vegan dishes")
        print("\n--- Assistant Menu Query ---")
        print(res)
        self.assertTrue(len(res) > 0)

    def test_assistant_order_query_with_id(self):
        assistant = SnackStackAssistant()
        res = assistant.ask("Track order ORD-201")
        print("\n--- Assistant Order Query (ORD-201) ---")
        print(res)
        self.assertIn("Butter Chicken", res)

    def test_assistant_hitl_interrupt(self):
        assistant = SnackStackAssistant()
        # 1. Ask query without order ID
        res1 = assistant.ask("Where is my order?")
        print("\n--- Assistant HITL Step 1 (No ID) ---")
        print(res1)
        self.assertIn("❓", res1)

        # 2. Provide order ID on resume
        res2 = assistant.ask("ORD-201")
        print("\n--- Assistant HITL Step 2 (Resumed with ORD-201) ---")
        print(res2)
        self.assertIn("Butter Chicken", res2)

    def test_assistant_parallel_dispatch(self):
        assistant = SnackStackAssistant()
        res = assistant.ask("Pizza options and track ORD-203")
        print("\n--- Assistant Parallel Query (Menu + Order) ---")
        print(res)
        self.assertTrue("Pizza" in res or "Margherita" in res)
        self.assertTrue("Cheeseburger" in res or "203" in res or "Preparing" in res)


if __name__ == "__main__":
    unittest.main()
