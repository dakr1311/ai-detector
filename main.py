import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS so Flutter can talk to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models (Using absolute paths for Render stability)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'ai_text_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("AI Models Loaded Successfully")
except Exception as e:
    print(f"Error Loading Models: {e}")

class TextRequest(BaseModel):
    text: str

# 1. Health Check Route (Fixes 404 when visiting URL in browser)
@app.get("/")
def home():
    return {"status": "Server is Online", "usage": "POST to /predict"}

# 2. Prediction Route
@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        # STEP 1: Vectorize
        data_vectorized = vectorizer.transform([request.text])
        
        # STEP 2: Predict
        prediction = model.predict(data_vectorized)[0]
        
        # STEP 3: Label
        # Adjust based on your model: 1=AI, 0=Human (usually)
        label = "AI Generated Content" if prediction == 1 else "Human Written Content"
        
        return {"result": label}
    except Exception as e:
        return {"result": f"Processing Error: {str(e)}"}

if __name__ == "__main__":
    # Use dynamic port for Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
