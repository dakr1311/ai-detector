import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI()

# 2. Enable CORS (Vital for Flutter apps to connect to the cloud)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Securely load your models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
model_path = os.path.join(BASE_DIR, 'ai_text_model.pkl')

try:
    with open(tfidf_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("SUCCESS: TF-IDF and AI Model loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

class TextRequest(BaseModel):
    text: str

# 4. Root Route (Check this in your browser to avoid 404)
@app.get("/")
def check_status():
    return {"status": "Server is Online", "endpoint": "/predict"}

# 5. Prediction Logic
@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please provide text.")

    try:
        # STEP 1: Text goes into TF-IDF Vectorizer
        # This converts your words into a numerical matrix
        vectorized_input = vectorizer.transform([request.text])
        
        # STEP 2: Matrix goes into AI Text Model
        # The Logistic Regression model classifies the numbers
        prediction = model.predict(vectorized_input)[0]
        
        # STEP 3: Return human-readable result
        # Check your training labels (usually 1 is AI, 0 is Human)
        result_label = "AI Generated" if prediction == 1 else "Human Written"
        
        return {"result": result_label}
    
    except Exception as e:
        return {"result": f"Error: {str(e)}"}

if __name__ == "__main__":
    # Uses Render's dynamic port or defaults to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
