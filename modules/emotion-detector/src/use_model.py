#--------------------------------------------------------------------------------------------------------------
# load the tflite model.
# acquire the images from camera
# convert image: to grayscale, find faces, quantize
# perform inference
# show image with drawn rectangle and detected emotion
#
# TODO:  move preprocess_to_int8, infer_int8,emotion_dict to another module?
# ----------------------------------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import cv2


#---------use the quantized tflite model  MOVE THIS TO SOME OTHER FILE? As this is used here and for the pi as well
interpreter = tf.lite.Interpreter(model_path="../models/model_int8.tflite", num_threads=4)
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

# --- OpenCV setup ---
cv2.ocl.setUseOpenCL(False)
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # load once
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        inp_q = preprocess_to_int8(roi_gray)
        probs = infer_int8(inp_q)
        cls = int(np.argmax(probs))
        label = emotion_dict.get(cls, "Unknown")
        cv2.putText(frame, label, (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Video (INT8 TFLite)', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()