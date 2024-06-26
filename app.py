import cv2
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import base64
import tempfile
import os
from flask_mail import Mail, Message
import eventlet

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = 'demo@gmail.com'
MAIL_PASSWORD = 'demopassword123'

# Initialize Flask application
app = Flask(__name__)
app.config.from_mapping(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAIL_SERVER=MAIL_SERVER,
    MAIL_PORT=MAIL_PORT,
    MAIL_USE_TLS=MAIL_USE_TLS,
    MAIL_USE_SSL=MAIL_USE_SSL,
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
mail = Mail(app)

# Load the YOLOv8 model
model = YOLO('models/best.pt')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/send_email', methods=['POST'])
def send_email():
    """Handle email sending."""
    json_data = request.get_json()
    name, email, message = json_data['name'], json_data['email'], json_data['message']

    sender = app.config['MAIL_USERNAME']
    recipients = [sender]
    msg = Message(subject=name, sender=sender, recipients=recipients,
                  body=f"Email: {email}\nMessage: {message}")

    try:
        mail.send(msg)
        return jsonify({"message": "Email sent successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Handle image uploads and perform object detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('image')
    results_list = [process_image(file) for file in files if file.filename]

    return jsonify(results_list), 200

def process_image(file):
    """Process a single image for object detection."""
    try:
        file_path = save_uploaded_file(file)
        image = cv2.imread(file_path)
        results = model(image)
        report, image_base64 = generate_report_and_image(results, image, file.filename)
        return {'report': report, 'image': image_base64}
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def save_uploaded_file(file):
    """Save uploaded file to a temporary directory."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return file_path

def generate_report_and_image(results, image, filename):
    """Generate detection report and encode image."""
    report = f"Detection Report for '{filename}':\n"
    object_count = 0

    for detection in results:
        for index, detected in enumerate(detection.boxes.xyxy):
            x_min, y_min, x_max, y_max = detected[:4]
            bbox_coords = f"({x_min:.4f}, {y_min:.4f}) to ({x_max:.4f}, {y_max:.4f})"
            confidence_str = f"{detection.boxes.conf[index] * 100:.2f}%"

            report += f"  Hotspot {object_count + 1}:\n"
            report += f"    - Confidence: {confidence_str}\n"
            report += f"    - Bounding Box: {bbox_coords}\n"
            object_count += 1

            draw_bounding_box(image, x_min, y_min, x_max, y_max, confidence_str)

    if object_count == 0:
        report += "  No Hotspots detected.\n"

    ret, jpeg = cv2.imencode('.jpg', image)
    if not ret:
        raise ValueError('Error encoding image')

    image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    return report, image_base64

def draw_bounding_box(image, x_min, y_min, x_max, y_max, confidence_str):
    """Draw bounding box and confidence on image."""
    pt1, pt2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(image, confidence_str, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle real-time video frames for object detection."""
    try:
        frame = decode_image(data)
        results = model(frame)
        frame = draw_detections(results, frame)
        frame_encoded = encode_image(frame)
        emit('processed_frame', frame_encoded)
    except Exception as e:
        print(f"Error processing frame: {e}")

def decode_image(data):
    """Decode base64 image to NumPy array."""
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(image):
    """Encode image to base64."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def draw_detections(results, frame):
    """Draw detections on the frame."""
    for detection in results:
        for detected in detection.boxes.xyxy:
            x_min, y_min, x_max, y_max = detected[:4]
            draw_bounding_box(frame, x_min, y_min, x_max, y_max, f"{detection.boxes.conf[0] * 100:.2f}%")
    return frame

if __name__ == '__main__':
    socketio.run(app, debug=True)

    