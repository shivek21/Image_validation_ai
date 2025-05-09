import mediapipe as mp
import cv2 
import numpy as np 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 🧠 3. Eye Detection from Image
# ===============================
def detect_eye_status_from_image(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    
    results = face_mesh.process(image)
    is_expression = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            
            # Extract landmark positions
            left_eye = [np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]) for i in LEFT_EYE]
            right_eye = [np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]) for i in RIGHT_EYE]

            # Compute EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            if(avg_ear < 0.21):
                is_expression = False

            return is_expression  
            
