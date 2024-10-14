from face_recognition import calculate_similarity

database = {}


def add_face(name, embedding):
    """Add a new face to the database."""
    database[name] = embedding


def find_closest_match(embedding, threshold=0.6):
    """Find the closest face match in the database."""
    closest_name = None
    highest_similarity = 0
    for name, stored_embedding in database.items():
        similarity = calculate_similarity(embedding, stored_embedding)
        if similarity > highest_similarity and similarity > threshold:
            highest_similarity = similarity
            closest_name = name
    return closest_name, highest_similarity
