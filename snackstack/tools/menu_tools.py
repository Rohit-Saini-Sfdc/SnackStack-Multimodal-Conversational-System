from langchain_core.tools import tool
from snackstack.tools.rag import search_menu


@tool
def search_menu_catalog(query: str) -> str:
    """Searches the SnackStack menu catalog for dishes matching a query, price range, cuisine, or dietary preference.

    Args:
        query: The search string or requirement (e.g., 'vegan options', 'Indian under 300', 'pizza', etc.)

    Returns:
        A detailed string of matching menu items.
    """
    docs = search_menu(query, k=5)
    if not docs:
        return "No matching dishes found in the menu catalog."

    results = []
    for doc in docs:
        results.append(doc.page_content)

    return "\n---\n".join(results)
