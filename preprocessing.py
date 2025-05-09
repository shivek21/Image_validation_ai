import cv2
import logging

def preprocess_image(image, target_width=224, target_height=224):
    """
    Preprocesses the input image by resizing and converting it to grayscale.
    """
    if image is None:
        logging.error("Input image is None")
        raise ValueError("Input image is None")

    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    logging.info(f"Image preprocessed: Resized to {target_width}x{target_height}")
    return grayscale_image, resized_image_rgb