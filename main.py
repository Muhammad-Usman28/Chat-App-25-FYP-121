from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
 
app = FastAPI()
 
# CORS Middleware (if required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as per your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Dataset Path
dataset_path = os.path.join(os.getcwd(), 'all_hadiths_clean.csv')
 
# Data Loading and Preprocessing
data = pd.read_csv(dataset_path)
 
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text
 
data['clean_text'] = data['text_en'].apply(preprocess_text)
data = data.dropna(subset=['clean_text']).reset_index(drop=True)
 
# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['clean_text'])
 
# Request Model
class Query(BaseModel):
    query: str
 
# Utility Function
def get_similar_hadees(query, tfidf_matrix, tfidf_vectorizer, top_n=10):
    query_vector = tfidf_vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_hadees_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    similar_hadees_scores = cosine_similarities[similar_hadees_indices]
    similar_hadees = data.iloc[similar_hadees_indices][['hadith_no', 'text_en', 'source']]
    return similar_hadees, similar_hadees_scores
 
# Routes
@app.post("/get_similar_hadees")
async def simple_post(query: Query):
    try:
        if not query.query:
            raise HTTPException(status_code=400, detail="Query is required.")
       
        similar_hadees, similar_hadees_scores = get_similar_hadees(query.query, tfidf_matrix, tfidf_vectorizer)
        result = []
        for _, row in similar_hadees.iterrows():
            hadees_info = {
                "hadith_no": row['hadith_no'],
                "source": row['source'],
                "text_en": row['text_en'],
            }
            result.append(hadees_info)
 
        return {"similar_hadees": result}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/getresult")
async def get_result():
    return {"similar_hadees": "Good"}

@app.get("/")
def read_root():
    return {"message":"Welcome to Islamic Chat App"}
