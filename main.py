from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import uvicorn

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load your uploaded models
# Make sure these files are in the same folder as main.py
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('ai_text_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

# 3. Define the request structure (Matches your Flutter jsonEncode)
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # 4. Process: Transform the text exactly like training
        vectorized_text = vectorizer.transform([request.text])
        
        # 5. Predict
        prediction = model.predict(vectorized_text)[0]
        
        # 6. Map result (Adjust 'AI' or 'Human' based on your model's labels)
        # Assuming 1 = AI and 0 = Human. Double-check your training labels!
        result_label = "AI Generated" if prediction == 1 else "Human Written"
        
        return {"result": result_label}
    
    except Exception as e:
        return {"result": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)