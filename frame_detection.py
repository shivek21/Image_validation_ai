import mediapipe as mp
def is_face_center_aligned(resized_image_rgb):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(resized_image_rgb)
    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = resized_image_rgb.shape

    xmin = int(bbox.xmin * w)
    ymin = int(bbox.ymin * h)
    box_width = int(bbox.width * w)
    box_height = int(bbox.height * h)

    # ✅ Use bounding box center instead of top-left
    x_center = xmin + box_width // 2
    y_center = ymin + box_height // 2

    top_cut = ymin < 0.05 * h
    bottom_cut = (ymin + box_height) > 0.95 * h
    centered = (0.35 * w < x_center < 0.65 * w) and (0.35 * h < y_center < 0.65 * h)

    is_framing=False
    if not centered:
        return is_framing
    elif top_cut or bottom_cut:
        return is_framing
    else:
        is_framing=True
        return is_framing
