#vector search

import numpy as np 
import pickle
from typing import List,Dict 
from .vector_store import get_embedding,load_faiss_index

#perform retrievals i.e vector search over product embeddings 
import numpy as np
from typing import List, Dict

def search_products(
    query: str,
    index,
    products: List[dict],
    k: int = 5
) -> List[Dict]:

    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    seen = set()
    results = []

    if len(products) == 0:
        return []

    for idx, score in zip(indices[0], distances[0]):

        if idx == -1:
            continue

        if idx >= len(products):
            continue

        product = products[idx]

        if not isinstance(product, dict):
            print("BROKEN PRODUCT TYPE:", type(product))
            print("VALUE:", product)
            continue

        name = product.get("product_name")

        if name in seen:
            continue

        seen.add(name)

        enriched = product.copy()
        enriched["similarity_score"] = float(score)

        results.append(enriched)

    return results