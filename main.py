# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()

# Load embedding and summarization models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Connect to ChromaDB collection
client = PersistentClient(path="./chroma_db")
collection = client.get_collection("books")

# Request models with example data for Swagger UI
class QueryRequest(BaseModel):
    query: str = Field(..., example="mysterious and thrilling")
    n_results: int = Field(5, example=3)

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        example="In a dystopian future, society is controlled by technology and memories are traded like currency. A young hacker must navigate a dangerous world."
    )

@app.post("/recommend")
def recommend(request: QueryRequest):
    query_embedding = embed_model.encode(request.query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results,
        include=["documents", "metadatas"]
    )
    recommendations = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        recommendations.append({
            "title": meta["Title"],
            "authors": meta["Authors"],
            "category": meta["Category"],
            "description": doc
        })
    return {"results": recommendations}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    summary = summarizer(request.text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}
