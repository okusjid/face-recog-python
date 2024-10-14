# app/main.py
from fastapi import FastAPI, UploadFile, File
from face_recognition import detect_faces, get_face_embedding
from database import add_face, find_closest_match
import cv2
import numpy as np

app = FastAPI()


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
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
