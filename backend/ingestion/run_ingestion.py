import pandas as pd
from backend.retrieval.vector_store import build_faiss_index, save_faiss_index


# load products
df = pd.read_csv("D:\\Ecommerce assistant using RAG\\data\\amazon.csv")
products = df.to_dict(orient="records")


print(products[0])

# build index
index, products = build_faiss_index(products)

# save index + metadata
save_faiss_index(index, products)

print("FAISS index and product metadata saved successfully.")
