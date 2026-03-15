#llm context building 
#Take reranked products and build a clean, factual, bounded context.
#Ingestion → Embedding → FAISS → Search → Rerank → Context → LLM


from typing import List, Dict, Optional

def build_context(
    products: List[Dict],
    image_description: Optional[str] = None
) -> str:
    """Builds the text context for the LLM using retrieved product data."""

    if not products:
        return "No relevant products were found."

    context_lines = []

    if image_description:
        context_lines.append(f"Image description:\n{image_description}\n")

    for idx, product in enumerate(products, start=1):
        context = (
            f"Product {idx}:\n"
            f"Name: {product['product_name']}\n"
            f"Category: {product['category']}\n"
            f"Price: {product['actual_price']} Rs\n"
            f"Description: {product['about_product']}\n"
        )
        context_lines.append(context)

    return "\n".join(context_lines)
