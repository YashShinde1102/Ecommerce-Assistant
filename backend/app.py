from fastapi import FastAPI,UploadFile,File,Form
from fastapi.middleware.cors import CORSMiddleware
import shutil 
import os 
from typing import Optional 
from backend.pipeline.run_pipeline import run_pipeline

app=FastAPI(title="Multimodel Ecommerce RAG API ")

#allow front end 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #later restrict the origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER="temp_upload"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.post("/query")
async def query_product(
    user_query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    image_path = None

    if image:
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    answer = run_pipeline(user_query=user_query, image_path=image_path)

    return {
        "query": user_query,
        "image_used": bool(image),
        "answer": answer
    }