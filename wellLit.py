import cv2
import numpy as np

def is_lighting_correct(image, min_brightness=60, max_brightness=200):
    """
    Check if the lighting in the image is within the acceptable range.
    
    :param image: Input image (numpy array).
    :param min_brightness: Minimum acceptable brightness.
    :param max_brightness: Maximum acceptable brightness.
    :return: True if lighting is correct, False otherwise.
    """
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the average brightness
    avg_brightness = np.mean(grayscale_image)
    
    # Check if the brightness is within the acceptable range
    return min_brightness <= avg_brightness <= max_brightness