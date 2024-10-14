from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (ensure you replace this with your model path)
MODEL = load_model("YOUR MODEL DIRECTORY GOES HERE")

# Class names (ensure these match your model's output classes)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def predict_image(img: Image.Image):
    img = img.resize((256, 256))  # Resize as per model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = MODEL.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    
    return predicted_class, confidence

# Endpoint to upload and predict
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file)
    predicted_class, confidence = predict_image(img)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
