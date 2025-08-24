#------------------------------------------------------------------------------
# load the tflite model - base or personalized
# take image from the camera module
# convert to grayscale, find face/s, quantize, run inference
# display image with a rectangle on and detected emotion
# show it on the webserver
#
# usage:
# python3 ~/projects/tinyml/emotions/pi_stream.py --model [base/personalized]
# if parameter not specified - it will load the base model
#------------------------------------------------------------------------------


import cv2, numpy as np, threading, time
from flask import Flask, Response


from tflite_runtime.interpreter import Interpreter
from pathlib import Path
from picamera2 import Picamera2

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--model")
model_type = ap.parse_args().model


#clahe = cv2.createCLAHE(/clipLimit = 2.0, tileGridSize = (8,8))

#def put_hud(img, faces_count):
#    cv2.putText(img, f"faces:{faces_count}", (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


DETECT_EVERY = 3

cascade_path=Path(__file__).parent/'haarcascade_frontalface_default.xml'
facecasc=cv2.CascadeClassifier(str(cascade_path))

emotion_dict = {
    0:"Angry", 1:"Disgusted", 2:"Fearful",
    3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"
}


#---------use the model
if model_type == "personalized":
    model_name = "model_int8_personalized.tflite"
else: 
#model_type == "base":
    model_name = "model_int8.tflite"

#interpreter = Interpreter(model_path="/home/pi/projects/tinyml/emotions/model_int8.tflite", num_threads=4)
m_path = os.path.join("/home/pi/projects/tinyml/emotions/", model_name)
interpreter = Interpreter(model_path=m_path, num_threads=4)
interpreter.allocate_tensors()

in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# Input / output specs
_, in_h, in_w, in_c = in_det["shape"] 
in_dtype  = in_det["dtype"] 
in_scale, in_zp = in_det.get("quantization", (0.0, 0))

out_dtype = out_det["dtype"]
out_scale, out_zp = out_det.get("quantization", (0.0, 0))

assert in_dtype in (np.int8, np.uint8), f"Expected INT8/UINT8 model, got {in_dtype}"

# Labels
emotion_dict = {
    0:"Angry", 1:"Disgusted", 2:"Fearful",
    3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"
}

def preprocess_to_int8(roi_gray):
    """Resize -> float32 [0,1] -> quantize to int8/uint8 as the model expects."""
    x = cv2.resize(roi_gray, (in_w, in_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    x /= 255.0  # match training rescale
    x = x[np.newaxis, :, :, np.newaxis]  # (1,H,W,1)
    # quantize: q = real/scale + zero_point
    scale = in_scale if in_scale != 0 else 1.0
    q = np.round(x / scale + in_zp)
    q = np.clip(q, np.iinfo(in_dtype).min, np.iinfo(in_dtype).max).astype(in_dtype)
    return q

def infer_int8(inp_q):
    interpreter.set_tensor(in_det["index"], inp_q)
    interpreter.invoke()
    out_q = interpreter.get_tensor(out_det["index"])[0]
    # dequantize to float for readability; argmax would work on int too
    if out_scale and (out_dtype in (np.int8, np.uint8)):
        out_f = (out_q.astype(np.float32) - out_zp) * out_scale
    else:
        out_f = out_q.astype(np.float32)
    return out_f


#---------camera
picam2=Picamera2()
cfg = picam2.create_preview_configuration(main={"format":"XRGB8888", "size":(1280,720)})
picam2.configure(cfg)
picam2.start()

class Streamer:
    def __init__(self,):
        # streaming state
        self.lock = threading.Lock()
        self.jpeg_frame = None
        self.running = True
        self.faces = []
        self.frame_id = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    

    def _loop(self):
        cv2.ocl.setUseOpenCL(False)

        while self.running:
            frame = picam2.capture_array()
            if frame is None:
                time.sleep(0.005)
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


            if self.frame_id % DETECT_EVERY == 0:
                self.faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in self.faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                inp_q = preprocess_to_int8(roi_gray)
                probs = infer_int8(inp_q)
                cls = int(np.argmax(probs))
                label = emotion_dict.get(cls, "Unknown")
                ty=max(10, y-10)
#                cv2.putText(frame, label, (x+20, y-60),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            
                cv2.putText(frame, label, (x+5, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

#            put_hud(frame, len(self.faces))

            # Encode to JPEG for MJPEG streaming
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                with self.lock:
                    self.jpeg_frame = buf.tobytes()

            self.frame_id += 1

    def get_jpeg(self):
        with self.lock:
            return self.jpeg_frame

    def stop(self):
        self.running = False
        self.t.join(timeout=2)


streamer = Streamer()


# --- Flask app ---
app = Flask(__name__)

@app.route("/")
def index():
    # simple viewer page
    return """
    <html><head><title>Emotion Stream</title></head>
    <body style="margin:0;background:#111;">
      <div style="text-align:center;">
        <h2 style="color:#eee;font-family:sans-serif;">Emotion Detection</h2>
        <img src="/video_feed" style="max-width:96vw;height:auto;border:0;"/>
      </div>
    </body>
    </html>
    """

def mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = streamer.get_jpeg()
        if frame is None:
            time.sleep(0.01)
            continue
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try:
        # 0.0.0.0 -> accessible from other devices on LAN
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        streamer.stop()
        picam2.stop()
