from datetime import datetime
from imutils.video import VideoStream
from argparse import ArgumentParser
from ImageDetection.MotionDetection.MotionDetector import MotionDetector
from flask import Response, Flask, render_template, request, abort
import requests
import time
import cv2
import imutils
import threading
# line bot sdk
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
# env settings
import env
import user_list

output_frame = None
lock = threading.Lock()
app = Flask(__name__)
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

line_bot_api = LineBotApi(env.ACCESS_TOKEN)
handler = WebhookHandler(env.SECRET_KEY)
user_id = ''


@app.route("/callback", methods=['POST'])
def callback():
    global user_id

    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    body_asJson = request.get_json()
    user_id = body_asJson['events'][0]['source']['userId']
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@app.route("/")
def index():
    # get parameters from a URL
    code = request.args.get('code')
    if get_user_id(code) == user_list.users['test']:
        return render_template("index.html")
    else:
        return Response('Access Deny')


def get_user_id(code):
    data_as_dict = {'Content-Type': 'application/x-www-form-urlencoded',
                    'grant_type': 'authorization_code',
                    'code': code,
                    'redirect_uri': env.REDIRECT_URI,
                    'client_id': env.CHANNEL_ID,
                    'client_secret': env.CLIENT_SECRET
                    }
    # get Access token
    req = requests.post(env.GET_ACCESS_TOKEN_URL, data=data_as_dict).json()
    id_toekn = req["id_token"]
    # get user profile
    data_as_dict = {'id_token':id_toekn, 'client_id':env.CHANNEL_ID}
    req = requests.post(env.ENDPOINT, data=data_as_dict).json()

    return req['sub']


def detect_motion(frame_count):
    """Detect and mark moving objects

    Args:
        frame_count (int): number of video frame
    """
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
    """ translate frames to image (byte)

    Yields:
        byte: Content-Type is image/jpeg
    """
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
    detect_motion_thread = threading.Thread(
        target=detect_motion, args=(args["frame_count"],))
    detect_motion_thread.daemon = True
    detect_motion_thread.start()
    # Do not auto reload for webcam
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True,
            use_reloader=False)

video_stream.stop()
