#FAISS index logic
import pickle
import os

INDEX_PATH = "data/vector_store/index.faiss"
META_PATH = "data/vector_store/products.pkl"
import faiss
import numpy as np 
from typing import List,Dict 
import google.genai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

#initialize model 
"""client = genai.Client()"""

"""def get_embedding(text:str)->np.ndarray:  #the input text must be string which will return numpy array 
  if not isinstance(text,str):
    raise TypeError("text must be a string")
  text=text.strip()
  if not text:
    raise ValueError("empty text cannot be embedded")
  
  response=client.models.embed_content(
    model="models/embedding-001",
    contents=text
  )

  vector=response.embeddings[0].values
  return np.array(vector,dtype=np.float32)
"""
#embeddings using the sentence transformers 
#offline and fast 

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        raise ValueError("Empty text")

    embedding = _embedding_model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)

def build_faiss_index(products):
    vectors = []
    clean_products = []

    for product in products:
        text = (
            f"{product['product_name']} "
            f"{product['about_product']} "
            f"{product['category']}"
        ).strip()

        if not text:
            continue

        embedding = get_embedding(text)
        vectors.append(embedding)
        clean_products.append(product)

    vectors = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, clean_products
 #tracks index position and map it back to product data 

def save_faiss_index(index, products):
    os.makedirs("data/vector_store", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(products, f)


def load_faiss_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("FAISS index or metadata not found. Run ingestion first.")

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        products = pickle.load(f)

    return index, products
 