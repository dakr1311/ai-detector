import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Required for Flutter
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI()

# 2. ADD CORS (This prevents many connection errors from Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load your uploaded models
# Note: Using absolute path handling for cloud servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'ai_text_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models: {e}")

# 4. Define the request structure
class TextRequest(BaseModel):
    text: str

# 5. Home Route (Use this to check if server is alive in a browser)
@app.get("/")
def read_root():
    return {"status": "Server is Online"}

# 6. Predict Route
@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Transform and Predict
        vectorized_text = vectorizer.transform([request.text])
        prediction = model.predict(vectorized_text)[0]
        
        # Labeling (Verify if 1=AI or 0=AI in your specific training)
        result_label = "AI Generated" if prediction == 1 else "Human Written"
        
        return {"result": result_label}
    
    except Exception as e:
        return {"result": f"Error during processing: {str(e)}"}

if __name__ == "__main__":
    # IMPORTANT: Render provides a $PORT environment variable. 
    # This line reads that variable or defaults to 8000.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
