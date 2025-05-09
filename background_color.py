import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
def  is_background_color_correct(resized_img_rgb,target_color, tolerance=78):
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = os.path.join("F:\\Franciscan\\image_Validation_model", "deeplab_v3.tflite")
    # Create a image segmenter instance with the image mode:
    # Read the model file as bytes
    with open(model_path, "rb") as f:
        model_content = f.read()
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model_content),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)
    

   

    # Wrap the image in a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_img_rgb)

    # Create an ImageSegmenter instance
    with ImageSegmenter.create_from_options(options) as segmenter:
        # Perform segmentation
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # Extract the background region
        background_mask = (category_mask == 0)  # Assuming 0 represents the background

        # Create a blank image for the background
        background_only = np.zeros_like(resized_img_rgb)

        # Copy only the background pixels from the original image
        background_only[background_mask] = resized_img_rgb[background_mask]

        # Extract only the non-black pixels from the background
        non_black_pixels = background_only[np.any(background_only != [0, 0, 0], axis=-1)]
        is_match= True
        # Check if there are any valid background pixels
        if non_black_pixels.size == 0:
            is_match = False
        # Calculate the mean color of the background
        mean_color = np.mean(non_black_pixels, axis=0)

        # Calculate the Euclidean distance between the mean color and the target color
        distance = np.linalg.norm(mean_color - np.array(target_color))
        # Check if the distance is within the tolerance
        is_match = distance <= tolerance

        # Return the result and the mean color
        return is_match