from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import os
from typing import Optional
import logging
import base64
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Skin Lesion Image Upload")

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    """Renderiza el formulario para subir imágenes"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analizador de Lesiones Cutáneas</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .upload-form { background-color: #f7f9fc; padding: 20px; border-radius: 8px; }
            .submit-btn { background-color: #3498db; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
            .submit-btn:hover { background-color: #2980b9; }
            .file-input { margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <h1>Analizador de Lesiones Cutáneas</h1>
        <div class="upload-form">
            <h2>Sube tu imagen para análisis</h2>
            <form action="/upload/" method="post" enctype="multipart/form-data">
                <div class="file-input">
                    <p>Selecciona una imagen de la lesión (.jpg, .jpeg o .png)</p>
                    <input type="file" name="file" accept=".jpg,.jpeg,.png" required>
                </div>
                <button type="submit" class="submit-btn">Analizar Imagen</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Leer bytes
    content = await file.read()
    b64data = base64.b64encode(content).decode()
    payload = {
        "filename": file.filename,
        "content_type": file.content_type,
        "data": b64data
    }
    resp = requests.post(
        "https://dyxpjb1lb8.execute-api.sa-east-1.amazonaws.com/prod2/upload",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)