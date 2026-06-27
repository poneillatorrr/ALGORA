from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort
import numpy as np
import io


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model
session = ort.InferenceSession("mobilenet_cancer_model.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


classes = [
    "benign",
    "malignant",
    "normal"
]


@app.get("/")
def home():
    return {
        "message": "Cancer classifier API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(
        io.BytesIO(await file.read())
    ).convert("RGB")


    # same preprocessing as your predict_onnx.py
    image = image.resize((224,224))

    img = np.array(image).astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)


    result = session.run(
        [output_name],
        {
            input_name: img
        }
    )


    prediction = result[0][0]

    index = np.argmax(prediction)

    confidence = float(prediction[index]) * 100


    return {
        "result": classes[index],
        "confidence": round(confidence,2)
    }