from langchain_core.tools import tool
from snackstack.data import ORDER_DATABASE


@tool
def get_order_status(identifier: str) -> str:
    """Retrieves the order status, item details, customer info, and delivery tracking for a given Order ID, Tracking ID, or Email address.

    Args:
        identifier: Order ID (e.g. ORD-201), Tracking ID (e.g. SS201TRK), or Email address (e.g. rohit@example.com).

    Returns:
        Formatted string containing order details or error message if not found.
    """
    clean_id = identifier.strip().lower()

    # 1. Check exact Order ID match
    for order_id, details in ORDER_DATABASE.items():
        if order_id.lower() == clean_id:
            return _format_order(details)

    # 2. Check Tracking ID match
    for details in ORDER_DATABASE.values():
        if details["tracking_id"].lower() == clean_id:
            return _format_order(details)

    # 3. Check Email match
    matched_orders = [
        details
        for details in ORDER_DATABASE.values()
        if details["customer_email"].lower() == clean_id
    ]
    if matched_orders:
        return "\n---\n".join([_format_order(o) for o in matched_orders])

    # 4. Partial substring search across Order ID, Tracking ID, or Email
    for details in ORDER_DATABASE.values():
        if (
            clean_id in details["order_id"].lower()
            or clean_id in details["tracking_id"].lower()
            or clean_id in details["customer_email"].lower()
        ):
            return _format_order(details)

    return f"No order found matching identifier: '{identifier}'. Please double check your Order ID, Tracking ID, or Email."


def _format_order(order: dict) -> str:
    return (
        f"Order ID: {order['order_id']}\n"
        f"Item: {order['item_name']}\n"
        f"Customer: {order['customer_name']} ({order['customer_email']})\n"
        f"Status: {order['status']}\n"
        f"Price: INR {order['price']}\n"
        f"Order Date: {order['order_date']}\n"
        f"Estimated Delivery: {order['estimated_delivery']}\n"
        f"Tracking ID: {order['tracking_id']}"
    )
