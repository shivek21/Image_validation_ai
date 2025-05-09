import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
def is_hair_on_forehead(resized_image_rgb):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path1 = os.path.join("F:\\Franciscan\\image_Validation_model", "face_landmarker.task")
    with open(model_path1, "rb") as f:
        model_content = f.read()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_content),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True
        )

    face_landmarker = FaceLandmarker.create_from_options(options)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
    result = face_landmarker.detect(image)

    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path=os.path.join("F:\\Franciscan\\image_Validation_model", "hair_segmenter.tflite")
# Create a image segmenter instance with the image mode:
    # Read the model file as bytes
    with open(model_path, "rb") as f:
        model_content1 = f.read()
    optionsHair = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model_content1),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)
    

    with ImageSegmenter.create_from_options(optionsHair) as segmenter:
        # Read the image from file
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)
    # Perform segmentation on the image
        segmentation_result = segmenter.segment(image)
    # Get the category mask for the hair class
        category_mask = segmentation_result.category_mask

    # Debug: Check the shape and unique values of the category mask

        category_mask_array = np.array(category_mask.numpy_view())


    forehead_indices = [10, 67, 69, 104, 108, 109, 151, 337, 338,9, 66, 68, 107, 336,105, 334]
    image_height, image_width = resized_image_rgb.shape[:2]

    # Convert forehead landmarks to pixel coordinates
    overlapping_indices = []
    face_landmarksresult = result.face_landmarks[0]
    for i in forehead_indices:
        landmark = face_landmarksresult[i]
        x_pixel = int(landmark.x * image_width)
        y_pixel = int(landmark.y * image_height)

            # Check if the pixel in the hair mask is 1 (hair region)
        if category_mask_array[y_pixel, x_pixel] == 1:
            overlapping_indices.append(i)
    isHair=False
    if len(overlapping_indices) > 0:
        isHair=True
        return isHair
    else:
        return isHair