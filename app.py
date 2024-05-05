import cv2
from ultralytics import YOLO
from flask import Flask, Response, request, jsonify
import tempfile
import os
import base64

# Flask APPLICATION
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load the YOLOv8 model
model = YOLO('models/best.pt')

# Route to handle image upload and object detection
@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to a temporary directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Perform object detection on the uploaded image
        try:
            report = ""
            image = cv2.imread(file_path)
            results = model(image)
            index_confidence = 0

            for detection in results: # Loop over detected objects
                for detected in detection.boxes.xyxy:    
                    x_min, y_min, x_max, y_max = detected[:4]   # Extract bounding box coordinates

                    confidence_str = detection.boxes.conf[index_confidence] * 100 # Confidence score
                    confidence_str = f'{confidence_str:.2f}%' # Format as percentage
                    index_confidence += 1 # Increment index

                    report += f"- Detected object: {confidence_str}\n"
                    report += f"  - Bounding box: ({x_min}, {y_min}) - ({x_max}, {y_max})\n"

                    # Create tuples for top-left and bottom-right corners
                    pt1 = (int(x_min), int(y_min))
                    pt2 = (int(x_max), int(y_max))

                    # Draw bounding box on the image using OpenCV
                    cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)  # Red color for better visibility

                    cv2.putText(image, str(confidence_str), pt1, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)  # White text, thick font 


            # Encode the processed image as JPEG
            ret, jpeg = cv2.imencode('.jpg', image)
            if not ret:
                return jsonify({'error': 'Error encoding image'})

            image_base64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            # Return the response with detection report and Base64-encoded image
            response_data = {
                'report': report,
                'image': image_base64
            }

            return jsonify(response_data), 200
            
        except Exception as e:
            # Handle image reading or processing errors
            return jsonify({'error': f'Error processing image: {str(e)}'})

        finally:
        
            # Release the temporary image file (if processing was successful)
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)


    