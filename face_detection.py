import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
def detect_faces(resized_image_rgb):
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = os.path.join("F:\\Franciscan\\image_Validation_model", "blaze_face_short_range.tflite")
# Create a face detector instance with the image mode:
# Read the model file as bytes
    with open(model_path, "rb") as f:
        model_content = f.read()
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_buffer=model_content),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.89)

    face_detector = FaceDetector.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
    # Detect faces in the image.
    detections = face_detector.detect(mp_image)
    num_faces = len(detections.detections)
    return num_faces
