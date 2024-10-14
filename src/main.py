# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .face_recognition import detect_faces, get_face_embedding
from .database import add_face, find_closest_match
import cv2
# import numpy as np  # Removed unused import

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    image_path = f"images/{file.filename}"
    try:
        with open(image_path, "wb") as image:
            content = await file.read()
            image.write(content)
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

    try:
        faces = detect_faces(image_path)
        if not faces:
            return {"error": "No faces detected"}
    except Exception as e:
        return {"error": f"Failed to detect faces: {str(e)}"}

    # For simplicity, assume only one face per image
    image = cv2.imread(image_path)
    key = list(faces.keys())[0]
    facial_area = faces[key]["facial_area"]
    embedding = get_face_embedding(image, facial_area)

    # Find the closest match from the database
    name, similarity = find_closest_match(embedding)
    if name:
        return {"name": name, "similarity": similarity}
    else:
        return {"message": "No match found. Please label the face."}


@app.post("/add_face/")
async def add_new_face(name: str, file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    image_path = f"images/{file.filename}"
    with open(image_path, "wb") as image:
        content = await file.read()
        image.write(content)

    # Detect faces in the image
    faces = detect_faces(image_path)
    if not faces:
        return {"error": "No faces detected"}

    # For simplicity, assume only one face per image
    image = cv2.imread(image_path)
    key = list(faces.keys())[0]
    facial_area = faces[key]["facial_area"]
    embedding = get_face_embedding(image, facial_area)

    # Add the face to the database
    if embedding is not None:
        add_face(name, embedding)
        return {"message": f"Face of {name} added successfully"}
    else:
        return {"error": "Failed to generate embedding"}
