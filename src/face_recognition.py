import cv2
import numpy as np
from retinaface import RetinaFace
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ArcFace (InsightFace) for embedding generation using CPU
face_embedder = FaceAnalysis(providers=["CPUExecutionProvider"])  # Use CPU provider
face_embedder.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU


def detect_faces(image_path):
    """Detect faces in the image using RetinaFace."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        faces = RetinaFace.detect_faces(img)
        if faces is None:
            raise ValueError("No faces detected in the image.")
        return faces
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return None


def get_face_embedding(image, facial_area):
    """Extract face embeddings using ArcFace."""
    try:
        x1, y1, x2, y2 = facial_area
        cropped_face = image[y1:y2, x1:x2]
        face = face_embedder.get(cropped_face)
        if face:
            return face[0].normed_embedding  # Get 512D vector
        else:
            raise ValueError("No face detected in the cropped area.")
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None


def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    try:
        if embedding1 is None or embedding2 is None:
            raise ValueError("One or both embeddings are None.")
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None
