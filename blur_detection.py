import cv2
import logging

def is_image_blurry(grayscale_image, threshold=272.0):
    """
    Detects if an image is blurry using the Laplacian Variance Method.

    Parameters:
        grayscale_image (numpy.ndarray): The grayscale image to analyze.
        threshold (float): The variance threshold to classify the image as blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
        float: The variance of the Laplacian.
    """
    if grayscale_image is None:
        logging.error("Grayscale image is None")
        raise ValueError("Grayscale image is None")

    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold

    print(f"Blur detection: Variance={variance}, Is blurry={is_blurry}")
    return is_blurry