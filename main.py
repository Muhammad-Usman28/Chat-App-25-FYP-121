from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset Path
dataset_path = os.path.join(os.getcwd(), 'all_hadiths_clean.csv')

# Load Dataset
data = pd.read_csv(dataset_path)

def preprocess_text(text):
    if pd.isnull(text) or text.strip() == "":
        return ""

    # Remove non-ASCII characters except Arabic script and commas
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF, ]', '', text)
    
    # Remove single quotes and parentheses
    text = re.sub(r"[\'`\(\)]", '', text)
    
    # Ensure only one space around commas
    text = re.sub(r'\s*,\s*', ', ', text)
    
    # Replace compound words like 'donot' into 'do not'
    contractions = {
        "donot": "do not", "wasnarrated": "was narrated",
        "cannot": "can not", "wont": "will not",
        "shouldnt": "should not", "wouldnt": "would not",
        "couldnt": "could not", "didnt": "did not",
        "cant": "can not", "aint": "are not",
        "doesnt": "does not", "isnt": "is not",
        "hasnt": "has not", "havent": "have not",
        "hadnt": "had not"
    }
    for key, value in contractions.items():
        text = re.sub(rf'\b{key}\b', value, text, flags=re.IGNORECASE)

    # Add space between concatenated words without proper spacing
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing to the dataset
data['clean_text'] = data['text_en'].apply(preprocess_text)
data = data.dropna(subset=['clean_text']).reset_index(drop=True)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['clean_text'])

# Request Model
class Query(BaseModel):
    query: str

# Function to Find Similar Hadees
def get_similar_hadees(query, tfidf_matrix, tfidf_vectorizer, top_n=10):
    query_vector = tfidf_vectorizer.transform([preprocess_text(query)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_hadees_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    similar_hadees = data.iloc[similar_hadees_indices][['hadith_no', 'text_en', 'source']]
    
    return similar_hadees

# API Endpoint
@app.post("/get_similar_hadees")
async def simple_post(query: Query):
    try:
        if not query.query:
            raise HTTPException(status_code=400, detail="Query is required.")
        
        similar_hadees = get_similar_hadees(query.query, tfidf_matrix, tfidf_vectorizer)
        result = []
        
        for _, row in similar_hadees.iterrows():
            hadees_info = {
                "hadith_no": preprocess_text(str(row['hadith_no'])),
                "source": preprocess_text(str(row['source'])),
                "text_en": preprocess_text(row['text_en']),
            }
            result.append(hadees_info)

        return {"similar_hadees": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to Islamic Chat App"}
