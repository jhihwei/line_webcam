# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

from datetime import datetime
from imutils.video import VideoStream
from argparse import ArgumentParser
from ImageDetection.MotionDetection.MotionDetector import MotionDetector
from flask import Response, Flask, render_template
import time
import cv2
import imutils
import threading


output_frame = None
lock = threading.Lock()
app = Flask(__name__)
video_stream = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frame_count):
    global video_stream, output_frame, lock
    motion_detector = MotionDetector(accum_weight=0.1)
    total = 0

    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=400)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        timestamp = datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_PLAIN, 0.8, (20, 20, 255), 1)
        if total > frame_count:
            motion = motion_detector.detect(gray_image)
            if motion is not None:
                (thresh, (min_X, min_y, max_x, max_y)) = motion
                cv2.rectangle(frame, (min_X, min_y),
                              (max_x, max_y), (0, 255, 0), 2)
        motion_detector.update(gray_image)
        total += 1
        with lock:
            output_frame = frame.copy()


def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    argparse_parser = ArgumentParser()
    argparse_parser.add_argument(
        "-i", "--ip", type=str, required=True, help="ip address of the device is require")
    argparse_parser.add_argument(
        "-p", "--port", type=int, required=True, help="port number of the server is require")
    argparse_parser.add_argument(
        "-f", "--frame-count", type=int, default=30, help="video frames")
    args = vars(argparse_parser.parse_args())
    detect_motion_thread = threading.Thread(target=detect_motion, args=(args["frame_count"],))
    detect_motion_thread.daemon = True
    detect_motion_thread.start()
    # Do not auto reload for webcam
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, 
    use_reloader=False)

video_stream.stop()
