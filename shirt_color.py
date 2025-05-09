import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
def is_shirt_color_correct(resized_image_rgb,target_color, tolerance=78):
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = os.path.join("F:\\Franciscan\\image_Validation_model", "selfie_multiclass_256x256.tflite")
    # Create a image segmenter instance with the image mode:
    # Read the model file as bytes
    with open(model_path, "rb") as f:
        model_content = f.read()
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model_content),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)

    # Wrap the image in a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_image_rgb)

    # Create an ImageSegmenter instance
    with ImageSegmenter.create_from_options(options) as segmenter:
        # Perform segmentation
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # Extract the background region
        shirt_mask = (category_mask == 4)  # Assuming 0 represents the background

        # Create a blank image for the background
        background_only = np.zeros_like(resized_image_rgb)

        # Copy only the background pixels from the original image
        background_only[shirt_mask] = resized_image_rgb[shirt_mask]

        # Extract only the non-black pixels from the background
        non_black_pixels = background_only[np.any(background_only != [0, 0, 0], axis=-1)]
        isShirt = False
        
        # Calculate the mean color of the background
        mean_color = np.mean(non_black_pixels, axis=0)

        # Calculate the Euclidean distance between the mean color and the target color
        distance = np.linalg.norm(mean_color - np.array(target_color))
        # Check if the distance is within the tolerance
        isShirt = distance <= tolerance

        # Return the result and the mean color
        return isShirt
    