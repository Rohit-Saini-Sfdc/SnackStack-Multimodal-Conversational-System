from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from snackstack.config import embeddings
from snackstack.data import MENU_CATALOG
from snackstack.logger import logger

_vector_store = None


def get_vector_store() -> Chroma:
    """Initializes and returns the Chroma vector store for the menu catalog."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    logger.info("Initializing ChromaDB menu vector store...")
    documents: List[Document] = []
    for item in MENU_CATALOG:
        dietary_str = ", ".join(item["dietary_tags"]) if item["dietary_tags"] else "None"
        content = (
            f"Dish: {item['name']}\n"
            f"Category: {item['category']}\n"
            f"Cuisine: {item['cuisine']}\n"
            f"Price: INR {item['price']}\n"
            f"Rating: {item['rating']}\n"
            f"Dietary: {dietary_str}\n"
            f"Description: {item['description']}\n"
            f"Available: {'Yes' if item['availability'] else 'No'}"
        )
        metadata = {
            "id": item["id"],
            "name": item["name"],
            "cuisine": item["cuisine"],
            "price": item["price"],
            "rating": item["rating"],
            "dietary": dietary_str,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    _vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="snackstack_menu",
    )
    logger.info(f"Loaded {len(documents)} dishes into vector store.")
    return _vector_store


def search_menu(query: str, k: int = 4) -> List[Document]:
    """Performs similarity search on the menu catalog."""
    store = get_vector_store()
    return store.similarity_search(query, k=k)
