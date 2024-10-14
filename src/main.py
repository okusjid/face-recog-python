# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .face_recognition import detect_faces, get_face_embedding
from .database import add_face, find_closest_match
import cv2

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


async def save_uploaded_file(file: UploadFile, path: str) -> str:
    try:
        with open(path, "wb") as image:
            content = await file.read()
            image.write(content)
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to save image: {str(e)}")


def process_image(image_path: str):
    try:
        faces = detect_faces(image_path)
        if not faces:
            raise ValueError("No faces detected")
        image = cv2.imread(image_path)
        key = list(faces.keys())[0]
        facial_area = faces[key]["facial_area"]
        embedding = get_face_embedding(image, facial_area)
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_path = f"images/{file.filename}"
    try:
        await save_uploaded_file(file, image_path)
        embedding = process_image(image_path)
        name, similarity = find_closest_match(embedding)
        if name:
            return {"name": name, "similarity": similarity}
        else:
            return {"message": "No match found. Please label the face."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/add_face/")
async def add_new_face(name: str, file: UploadFile = File(...)):
    image_path = f"images/{file.filename}"
    try:
        await save_uploaded_file(file, image_path)
        embedding = process_image(image_path)
        if embedding is not None:
            add_face(name, embedding)
            return {"message": f"Face of {name} added successfully"}
        else:
            return {"error": "Failed to generate embedding"}
    except Exception as e:
        return {"error": str(e)}
