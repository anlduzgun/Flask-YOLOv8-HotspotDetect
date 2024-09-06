# Flask YOLO Object Detection Application

This Flask application demonstrates real-time hotspot detection using the  custom trained YOLOv8 model. It also features functionalities for handling image uploads, performing object detection, sending emails, and processing real-time video frames through websockets.

## Features

- **Object Detection:** Upload images through the `/detect` endpoint to see the detected objects and their confidence levels.
![hotspot detected](/media/example.jpeg)

- **Real-Time Video Detection:** Connect to the WebSocket at `/video_frame` to stream video frames for real-time object detection.
![real time hotspot detection](/media/real-time-hotspot-detection.gif)

- **Send Email:** Use the `/send_email` endpoint to send emails. Modify the endpoint as needed to integrate with your frontend.


## Installation

Ensure you have Python 3.8+ installed, then clone this repository and navigate into the project directory.

```bash
git clone https://github.com/anlduzgun/Flask-YOLOv8-HotspotDetect.git
cd Flask-YOLOv8-HotspotDetect
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the application, configure the email settings in the `app.config` section of the `app.py` file to match your SMTP provider details.

## Running the Application

To start the server, run the following command:

```bash
python app.py
```

Access the application by navigating to `http://localhost:5000` in your web browser.


