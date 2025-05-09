from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import time  # Import the time module
from preprocessing import preprocess_image
from blur_detection import is_image_blurry
from face_detection import detect_faces
from frame_detection import is_face_center_aligned
from hair_detection import is_hair_on_forehead
from background_color import is_background_color_correct
from shirt_color import is_shirt_color_correct
from waitress import serve
import logging

app = Flask(__name__)

logging.basicConfig(
    filename="flask_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.route('/validate-image', methods=['POST'])
def validate_image():
    try:
        # Step 1: Get the image data from the request
        start_time = time.time()
        data = request.json
        base64_img = data.get('image_data')
        if not base64_img:
            return jsonify({"error": "No image data provided"}), 400
        logging.info(f"Step 1 completed in {time.time() - start_time:.2f} seconds")

        # Step 2: Decode the image data
        start_time = time.time()
        image_data = base64.b64decode(base64_img)
        image = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        logging.info(f"Step 2 completed in {time.time() - start_time:.2f} seconds")

        # Step 3: Preprocess the image
        start_time = time.time()
        grayscale_image, resized_image_rgb = preprocess_image(image)
        logging.info(f"Step 3 (Preprocessing) completed in {time.time() - start_time:.2f} seconds")

        # Step 4: Check if the image is blurry
        start_time = time.time()
        is_blurry = is_image_blurry(grayscale_image, threshold=272.0)
        logging.info(f"Step 4 (Blur Detection) completed in {time.time() - start_time:.2f} seconds")
        if is_blurry:
            return jsonify({"message": "The image is blurry"}), 200

        # Step 5: Detect faces in the image
        start_time = time.time()
        num_faces = detect_faces(image)
        logging.info(f"Step 5 (Face Detection) completed in {time.time() - start_time:.2f} seconds")
        print(f"Number of faces detected: {num_faces}")
        if num_faces == 0:
            return jsonify({"message": "The image contains no face"}), 200
        elif num_faces > 1:
            return jsonify({"message": "The image contains more than one face"}), 200

        # Step 6: Check if the face is center-aligned
        start_time = time.time()
        is_framing = is_face_center_aligned(image)
        logging.info(f"Step 6 (Face Alignment) completed in {time.time() - start_time:.2f} seconds")
        if not is_framing:
            return jsonify({"message": "The face is not center aligned"}), 200

        # Step 7: Check if there is hair on the forehead
        start_time = time.time()
        is_hair = is_hair_on_forehead(image)
        logging.info(f"Step 7 (Hair Detection) completed in {time.time() - start_time:.2f} seconds")
        if is_hair:
            return jsonify({"message": "The hair is on the forehead"}), 200

        # Step 8: Check if the background color is correct
        start_time = time.time()
        is_match = is_background_color_correct(image, target_color=[93, 116, 91], tolerance=78)
        logging.info(f"Step 8 (Background Color Check) completed in {time.time() - start_time:.2f} seconds")
        if not is_match:
            return jsonify({"message": "The background color is not the same as described"}), 200

        # Step 9: Check if the shirt color is correct
        start_time = time.time()
        is_shirt = is_shirt_color_correct(image, target_color=[185, 127, 32], tolerance=78)
        logging.info(f"Step 9 (Shirt Color Check) completed in {time.time() - start_time:.2f} seconds")
        if not is_shirt:
            return jsonify({"message": "The shirt color is not the described one"}), 200

        # If all checks pass
        logging.info("All checks passed successfully")
        return jsonify({"message": "Image verified successfully"}), 200

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    logging.info("Starting the Flask server...")
    serve(app, host="0.0.0.0", port=5000)