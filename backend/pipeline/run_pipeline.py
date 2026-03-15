from typing import Optional
from backend.vision.image_to_text import image_to_text
from backend.retrieval.search import search_products
from backend.retrieval.reranker import rerank_products
from backend.rag.context_builder import build_context
from backend.rag.answer_generator import generate_answer
from backend.retrieval.vector_store import load_faiss_index
import time 

index, products = load_faiss_index()

'''def run_pipeline(
    user_query: str,
    image_path: Optional[str] = None
) -> str:
   

    image_description = None

    if image_path:
        try:
            image_description = image_to_text(image_path)
            print("IMAGE DESCRIPTION:", image_description)
            final_query = f"""
 User question:
 {user_query}

 Image description:
{image_description}

        except Exception:
            final_query = user_query
    else:
        final_query = user_query

    retrieved_products = search_products(
        query=final_query,
        index=index,
        products=products
    )

    if not retrieved_products:
        return "I do not have enough information to answer that"

    reranked_products = rerank_products(retrieved_products)

    context = build_context(
        reranked_products,
        image_description=image_description
    )

    answer = generate_answer(context, final_query)
    return {answer}'''

def run_pipeline(
    user_query: str,
    image_path: Optional[str] = None
):
    import time 
    start_time=time.time()
    image_description = None

    if image_path:
        try:
            image_description = image_to_text(image_path)
            print("IMAGE DESCRIPTION:", image_description)
            enriched_query = f"""
Product description: {image_description}.
Identify the most semantically similar product from the catalog.
Focus on product type, category, and key attributes.
"""
            final_query = enriched_query
        except Exception:
            final_query = user_query
    else:
        final_query = user_query

    retrieved_products = search_products(
        query=final_query,
        index=index,
        products=products
    )
    reranked_products = rerank_products(final_query, retrieved_products)

    if not reranked_products:
     return {
        "image_caption": image_description,
        "matched_product": None,
        "price": None,
        "confidence": 0.0,
        "top_k_candidates": [],
        "llm_summary": "No match found."
     }

    
    

    # assume first product is best after reranking
    best_product = reranked_products[0]

    

    if best_product["confidence"] < 0.45:
     return {
        "image_caption": image_description,
        "matched_product": None,
        "price": None,
        "confidence": best_product["confidence"],
        "top_k_candidates": [],
        "llm_summary": "No confident match found."
      }

    context = build_context(
        [best_product],
        image_description=image_description
    )

    answer = generate_answer(context, final_query)

    print("Total pipeline time:", time.time() - start_time)

    return {
        "image_caption": image_description,
        "matched_product": best_product.get("product_name"),
        "price": best_product.get("actual_price"),
        "confidence_score":best_product.get("similarity_score"),
        "top_k_candidates": [
            {
                "product_name": p.get("product_name"),
                "price": p.get("actual_price"),
                "score":p.get("similarity_score")
            }
            for p in reranked_products[:3]

        ],
        "llm_summary": answer
    }



