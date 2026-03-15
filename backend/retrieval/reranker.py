from sentence_transformers import CrossEncoder
import numpy as np

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_products(query: str, products: list):

    if not products:
        return []

    pairs = []
    for p in products:
        text = (
            f"{p.get('product_name','')} "
            f"{p.get('about_product','')} "
            f"{p.get('category','')}"
        )
        pairs.append((query, text))

    rerank_scores = cross_encoder.predict(pairs)

    for p, score in zip(products, rerank_scores):
        p["rerank_score"] = float(score)

        # Hybrid score
        p["final_score"] = (
            0.6 * p["rerank_score"] +
            0.4 * p.get("similarity_score", 0)
        )

    # Softmax normalization
    scores = np.array([p["final_score"] for p in products])
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    for p, prob in zip(products, probs):
        p["confidence"] = float(prob)

    products = sorted(
        products,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return products