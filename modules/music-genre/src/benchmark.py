import time
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "../models/music_genre_lstm.tflite"   # CNN or LSTM float TFLite
AUDIO_PATH = "../test/c_1_2.wav"
MODEL_TYPE = "lstm"  # "cnn" or "lstm"

RUNS = 200          # total timed runs
WARMUP = 20         # warm-up runs (not timed)

# Audio / MFCC params (must match training!)
SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
FRAME_STEP = 1024
FFT_LENGTH = 2048
NUM_MFCCS = 18
NUM_MEL_FREQS = 40
FMIN_HZ = 20
FMAX_HZ = SAMPLE_RATE / 2
EXPECTED_FRAMES = 1 + int((SAMPLE_RATE - FRAME_LENGTH) // FRAME_STEP)

# -----------------------
# Load model
# -----------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
i_details = interpreter.get_input_details()[0]
o_details = interpreter.get_output_details()[0]

# -----------------------
# Load audio (1 second)
# -----------------------
audio, sr = sf.read(AUDIO_PATH)
if audio.ndim > 1:
    audio = audio[:, 0]
if sr != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
audio = audio[:SAMPLE_RATE].astype(np.float32)
# -----------------------
# MFCC extraction
# -----------------------
def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE,
        n_mfcc=NUM_MFCCS,
        n_fft=FFT_LENGTH,
        hop_length=FRAME_STEP,
        win_length=FRAME_LENGTH,
        center=False,
        n_mels=NUM_MEL_FREQS,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ
    )

    T = mfcc.shape[1]
    if T < EXPECTED_FRAMES:
        mfcc = np.pad(mfcc, ((0,0),(0, EXPECTED_FRAMES - T)))
    elif T > EXPECTED_FRAMES:
        mfcc = mfcc[:, :EXPECTED_FRAMES]

    return mfcc.astype(np.float32)

# -----------------------
# Inference wrapper
# -----------------------
def run_inference(mfcc):
    if MODEL_TYPE == "cnn":
        x = mfcc[np.newaxis, :, :, np.newaxis]   # (1,18,20,1)
    else:  # LSTM
        x = mfcc[np.newaxis, :, :]           # (1,18,20)
    interpreter.set_tensor(i_details["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(o_details["index"])[0]

# -----------------------
# Warm-up
# -----------------------
mfcc = extract_mfcc(audio)
for _ in range(WARMUP):
    run_inference(mfcc)

# -----------------------
# Benchmark
# -----------------------
mfcc_times = []
infer_times = []
total_times = []

for _ in range(RUNS):
    t0 = time.perf_counter()

    t1 = time.perf_counter()
    mfcc = extract_mfcc(audio)
    t2 = time.perf_counter()

    _ = run_inference(mfcc)
    t3 = time.perf_counter()

    mfcc_times.append((t2 - t1) * 1000)
    infer_times.append((t3 - t2) * 1000)
    total_times.append((t3 - t0) * 1000)

# -----------------------
# Report
# -----------------------
def stats(x):
    return np.mean(x), np.median(x), np.percentile(x, 95)

print("\n===== Raspberry Pi Benchmark =====")
print(f"Model: {MODEL_PATH}")
print(f"Type:  {MODEL_TYPE.upper()}")
print(f"Runs:  {RUNS}\n")

m, med, p95 = stats(mfcc_times)
print(f"MFCC time     : mean={m:.2f} ms | median={med:.2f} | p95={p95:.2f}")

m, med, p95 = stats(infer_times)
print(f"Inference time: mean={m:.2f} ms | median={med:.2f} | p95={p95:.2f}")

m, med, p95 = stats(total_times)
print(f"End-to-end    : mean={m:.2f} ms | median={med:.2f} | p95={p95:.2f}")

