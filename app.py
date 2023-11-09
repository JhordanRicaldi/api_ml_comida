from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import urllib.request
from PIL import Image
import io
import json

app = FastAPI()

# Cargar el modelo previamente exportado
model = load_model('modelo_entrenado.h5')

# Lista para almacenar los resultados
results = []

@app.post("/predict/")
async def predict(url: str = Form(...)):
    try:
        # Usa urllib para obtener la imagen de la URL
        with urllib.request.urlopen(url) as response:
            image_data = response.read()

        # Convierte la imagen en un array de numpy
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

        # Realizar la predicción
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalizar
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_predicted = np.argmax(prediction, axis=1)

        if class_predicted == 0:
            result = "La comida en la imagen es saludable."
        else:
            result = "La comida en la imagen no es saludable."

        # Agregar el resultado a la lista
        results.append({"url": url, "prediction": result})

        # Guardar los resultados en un archivo JSON
        with open("results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

        return JSONResponse(content={"prediction": result})
    except Exception as e:
        return JSONResponse(content={"error": "Error en la predicción", "details": str(e)})

@app.get("/results/")
async def get_results():
    return JSONResponse(content=results)

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="localhost", port=8000)
