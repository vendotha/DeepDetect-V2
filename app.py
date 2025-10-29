import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB Upload Limit

# --- Model & API Initialization ---
# Load Deepfake Detection Model
try:
    # Use the exact name of your downloaded model file
    deepfake_model = load_model('model/mesonet_model.h5', compile=False)
    print("✅ MesoNet model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading MesoNet model: {e}")
    deepfake_model = None

# Initialize Face Detector
try:
    face_detector = MTCNN()
    print("✅ MTCNN face detector loaded successfully.")
except Exception as e:
    print(f"❌ Error loading MTCNN: {e}")
    face_detector = None

# Configure Gemini API
# !! IMPORTANT: Set this in your environment, not in the code.
try:
    # This line reads the key from your terminal environment
    GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
    if not GOOGLE_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        gemini_model = None
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Use the vision model
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Your simple HTML page
@app.route('/')
def index():
    return render_template('index.html')


def preprocess_image(image_path):
    """
    Loads an image, detects a face, crops it, and resizes for MesoNet.
    Returns the processed face or None.
    """
    if not face_detector:
        return None, "Face detector not initialized"

    img = cv2.imread(image_path)
    if img is None:
        return None, "Could not read image"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(img_rgb)

    if not faces:
        return None, "No face detected in the image."

    # Use the first and most confident face
    x, y, w, h = faces[0]['box']

    # Ensure coordinates are positive
    x, y = abs(x), abs(y)

    # FIX: Crop from the RGB image, not the BGR one
    face_crop = img_rgb[y:y + h, x:x + w]

    if face_crop.size == 0:
        return None, "Face crop was empty."

    # MesoNet expects 256x256
    face_resized = cv2.resize(face_crop, (256, 256))
    face_resized = face_resized.astype('float32') / 255.0

    # Convert single image to a batch of 1
    return np.expand_dims(face_resized, axis=0), None


def get_gemini_explanation(image_path, prediction_label):
    """
    Sends the image to Gemini and asks for an explanation.
    """
    if not gemini_model:
        return "Gemini AI model is not configured. Please set GEMINI_API_KEY in your environment."

    try:
        img_pil = Image.open(image_path)

        prompt = f"""
                        This image was flagged as **{prediction_label}**.

                        Give me a quick, point-by-point analysis of the visual evidence. I need to know why it was flagged this way.

                        - Be direct, confident, and use natural language, like a human expert.
                        - Focus on specific details: skin, lighting, eyes, or edges.
                        - Keep it very short and easy for a non-expert to read.

                        Do not mention AI, models, or any formal analysis terms (like "artifact" or "compression"). Just give me the visual facts.
                        """

        response = gemini_model.generate_content([prompt, img_pil])
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not deepfake_model or not face_detector:
        return jsonify({'error': 'Server models not initialized. Check logs.'}), 500

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        file_type = filename.rsplit('.', 1)[1].lower()

        # --- Image Processing ---
        if file_type in {'png', 'jpg', 'jpeg'}:
            processed_face, error_msg = preprocess_image(filepath)

            if processed_face is None:
                os.remove(filepath)  # Clean up upload
                return jsonify({'error': error_msg}), 400

            # Run MesoNet prediction
            prediction = deepfake_model.predict(processed_face)[0][0]

            # --- LOGIC FIX ---
            # Assuming HIGH (near 1.0) = REAL
            prediction_label = "REAL" if prediction > 0.97 else "FAKE"
            confidence = f"{prediction * 100:.2f}%" if prediction_label == "REAL" else f"{(1 - prediction) * 100:.2f}%"

            # Get Gemini Explanation
            reason = get_gemini_explanation(filepath, prediction_label)

            os.remove(filepath)  # Clean up upload

            return jsonify({
                'filename': filename,
                'content_type': 'image',
                'prediction': prediction_label,
                'confidence': confidence,
                'reason': reason
            })

        # --- Video Processing ---
        elif file_type in {'mp4', 'mov'}:
            cap = cv2.VideoCapture(filepath)
            frame_predictions = []
            highest_fake_frame = None
            frame_to_explain = None
            max_fake_prob = 0.0  # Store highest fake prob (near 0.0)
            max_real_prob = 0.97  # Store highest real prob (near 1.0)

            frame_count = 0
            # Process 1 frame per second to save time
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1: fps = 25  # Default

            while cap.isOpened():
                frame_exists, frame = cap.read()
                if not frame_exists:
                    break

                # Check if this frame is one we should process
                frame_id = int(round(cap.get(1)))
                if frame_id % int(fps) == 0:
                    frame_count += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_detector.detect_faces(frame_rgb)

                    if faces:
                        x, y, w, h = faces[0]['box']
                        x, y = abs(x), abs(y)

                        # FIX: Crop from the RGB frame
                        face_crop = frame_rgb[y:y + h, x:x + w]

                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (256, 256))
                            face_resized = face_resized.astype('float32') / 255.0
                            face_batch = np.expand_dims(face_resized, axis=0)

                            pred = deepfake_model.predict(face_batch)[0][0]
                            frame_predictions.append(pred)

                            # Save the most "fake" looking frame for Gemini
                            if pred < (1 - max_real_prob):
                                max_real_prob = (1 - pred)  # Store as confidence, not raw score
                                frame_to_explain = frame

                            # Save the most "real" looking frame for Gemini
                            elif pred > max_real_prob:
                                max_real_prob = pred
                                frame_to_explain = frame

            cap.release()

            if not frame_predictions:
                os.remove(filepath)  # Clean up upload
                return jsonify({'error': 'No faces detected in any video frames.'}), 400

            # Aggregate results
            avg_prediction = np.mean(frame_predictions)

            # --- LOGIC FIX ---
            # Assuming HIGH (near 1.0) = REAL
            prediction_label = "REAL" if avg_prediction > 0.97 else "FAKE"
            confidence = f"{avg_prediction * 100:.2f}%" if prediction_label == "REAL" else f"{(1 - avg_prediction) * 100:.2f}%"

            # Save the frame-to-explain and send to Gemini
            reason = "Could not generate explanation for video."
            if frame_to_explain is not None:
                frame_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{filename}.jpg")
                cv2.imwrite(frame_filename, frame_to_explain)
                reason = get_gemini_explanation(frame_filename, prediction_label)
                os.remove(frame_filename)  # Clean up frame

            os.remove(filepath)  # Clean up video

            return jsonify({
                'filename': filename,
                'content_type': 'video',
                'frames_processed': frame_count,
                'prediction': prediction_label,
                'confidence': confidence,
                'reason': reason
            })

    return jsonify({'error': 'Invalid file type'}), 400


# --- Run the App ---
if __name__ == '__main__':
    # Make sure 'uploads' folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))