#ranking on the basis of the price/category
# Take retrieved products and apply hard constraints and simple logic that embeddings cannot capture.
#Query → Embed → Search → Rerank → Context → LLM
from typing import List, Dict, Optional

def rerank_products(
    products: List[Dict],
    max_price: Optional[float] = None,
    category: Optional[str] = None
) -> List[Dict]:
    """Apply rule-based ranking and filtering on retrieved products."""

    filtered = products

    if max_price is not None:
        filtered = [
            p for p in filtered
            if float(p["actual_price"]) <= max_price
        ]

    if category is not None:
        filtered = [
            p for p in filtered
            if p["category"] == category
        ]

    return filtered
