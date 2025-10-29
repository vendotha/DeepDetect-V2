import os
import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
# We no longer need all of tensorflow, just the 'lite' interpreter
import tflite_runtime.interpreter as tflite
from mtcnn.mtcnn import MTCNN
from PIL import Image

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB Upload Limit

# --- Model & API Initialization ---
# Load TFLite Deepfake Detection Model
try:
    interpreter = tflite.Interpreter(model_path='model/mesonet_model.tflite')
    interpreter.allocate_tensors()
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")
    interpreter = None

# Initialize Face Detector
try:
    face_detector = MTCNN()
    print("✅ MTCNN face detector loaded successfully.")
except Exception as e:
    print(f"❌ Error loading MTCNN: {e}")
    face_detector = None

# Configure Gemini API
try:
    GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
    if not GOOGLE_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        gemini_model = None
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✅ Gemini API configured.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


def tflite_predict(face_batch):
    """Helper function to run inference with the TFLite model."""
    if not interpreter:
        return 0.0  # Fail gracefully

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], face_batch)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return prediction[0][0]


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

    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)
    face_crop = img_rgb[y:y + h, x:x + w]

    if face_crop.size == 0:
        return None, "Face crop was empty."

    face_resized = cv2.resize(face_crop, (256, 256))

    # TFLite model expects float32
    face_resized = face_resized.astype('float32') / 255.0

    # Convert single image to a batch of 1 and ensure type
    face_batch = np.expand_dims(face_resized, axis=0).astype(np.float32)

    return face_batch, None


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

    if not interpreter or not face_detector:
        return jsonify({'error': 'Server models not initialized. Check logs.'}), 500

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure the uploads folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        file.save(filepath)

        file_type = filename.rsplit('.', 1)[1].lower()

        # --- Image Processing ---
        if file_type in {'png', 'jpg', 'jpeg'}:
            processed_face, error_msg = preprocess_image(filepath)

            if processed_face is None:
                os.remove(filepath)
                return jsonify({'error': error_msg}), 400

            # Run TFLite prediction
            prediction = tflite_predict(processed_face)

            # A HIGH score (near 1.0) means REAL
            # Use a threshold of 0.97 as we discovered
            prediction_label = "REAL" if prediction > 0.97 else "FAKE"
            confidence = f"{prediction * 100:.2f}%" if prediction_label == "REAL" else f"{(1 - prediction) * 100:.2f}%"

            reason = get_gemini_explanation(filepath, prediction_label)
            os.remove(filepath)

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

            frame_to_explain_real = None
            frame_to_explain_fake = None
            best_real_score = 0.0  # Look for high scores (max)
            best_fake_score = 1.0  # Look for low scores (min)

            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps < 1: fps = 25

            while cap.isOpened():
                frame_exists, frame = cap.read()
                if not frame_exists:
                    break

                frame_id = int(round(cap.get(1)))
                if frame_id % int(fps) == 0:
                    frame_count += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = face_detector.detect_faces(frame_rgb)

                    if faces:
                        x, y, w, h = faces[0]['box']
                        x, y = abs(x), abs(y)
                        face_crop = frame_rgb[y:y + h, x:x + w]

                        if face_crop.size > 0:
                            face_resized = cv2.resize(face_crop, (256, 256))
                            face_resized = face_resized.astype('float32') / 255.0
                            face_batch = np.expand_dims(face_resized, axis=0).astype(np.float32)

                            pred = tflite_predict(face_batch)
                            frame_predictions.append(pred)

                            if pred > best_real_score:
                                best_real_score = pred
                                frame_to_explain_real = frame.copy()

                            if pred < best_fake_score:
                                best_fake_score = pred
                                frame_to_explain_fake = frame.copy()

            cap.release()

            if not frame_predictions:
                os.remove(filepath)
                return jsonify({'error': 'No faces detected in any video frames.'}), 400

            avg_prediction = np.mean(frame_predictions)

            # A HIGH score (near 1.0) means REAL
            # Use a threshold of 0.97 as we discovered
            prediction_label = "REAL" if avg_prediction > 0.97 else "FAKE"
            confidence = f"{avg_prediction * 100:.2f}%" if prediction_label == "REAL" else f"{(1 - avg_prediction) * 100:.2f}%"

            reason = "Could not generate explanation for video."
            frame_to_save = None

            if prediction_label == "REAL":
                frame_to_save = frame_to_explain_real
            else:
                frame_to_save = frame_to_explain_fake

            if frame_to_save is not None:
                frame_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{filename}.jpg")
                cv2.imwrite(frame_filename, frame_to_save)
                reason = get_gemini_explanation(frame_filename, prediction_label)
                os.remove(frame_filename)

            os.remove(filepath)

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
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

