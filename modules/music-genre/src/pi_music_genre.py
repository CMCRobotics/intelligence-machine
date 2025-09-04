#------------------------------------------------------------------------------
# load the tflite model
# record from mic or open the song (located in "dataset" folder)
# calculate mfcc, quantize if intput in8 used, run inference
# play the loaded song
#
# usage:
# python3 ~/projects/tinyml/music_genre/pi_music_genre.py --mode [song_name]
# if parameter not specified - it will record from the mic
#------------------------------------------------------------------------------




from tflite_runtime.interpreter import Interpreter
import numpy as np
import sounddevice as sd
import librosa
import argparse
import soundfile as sf
import os
import pygame



SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
FRAME_STEP   = 1024
FFT_LENGTH   = 2048
FMIN_HZ = 20
FMAX_HZ = SAMPLE_RATE / 2
NUM_MEL_FREQS = 40
NUM_MFCCS = 18
LIST_GENRES = ['disco', 'jazz', 'metal']

# For 1 s audio: frames ~= 1 + floor((N - win)/hop)
# With 22050, win=2048, hop=1024 -> T ≈ 20
SECONDS = 1.0
EXPECTED_FRAMES = 1 + int((SAMPLE_RATE - FRAME_LENGTH) // FRAME_STEP)  # ≈ 20
HOME_DIR = "/home/pi/projects/tinyml/music_genre/"

#---------use the quantized tflite model
m_path = os.path.join(HOME_DIR, "music_genre.tflite")
interpreter = Interpreter(model_path=m_path) # num_threads=4)???
interpreter.allocate_tensors()

# Get input/output layer information
i_details = interpreter.get_input_details()[0]
o_details = interpreter.get_output_details()[0]

# Input / output specs
in_dtype  = i_details["dtype"]
in_scale, in_zp = i_details.get("quantization", (0.0, 0))

out_dtype = o_details["dtype"]
out_scale, out_zp = o_details.get("quantization", (0.0, 0))

# Expect input rank 3: [1, 18, 20] (batch, mfcc, frames)
print("TFLite input shape:", i_details["shape"])


def record_seconds(seconds=SECONDS, sr=SAMPLE_RATE):
    print(f"Recording {seconds:.2f} s @ {sr} Hz… (speak/play music now)")
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc_like_training(y):
    # librosa.feature.mfcc returns shape (n_mfcc, T)
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
    )  # shape (18, T)

    # Pad/trim to exactly EXPECTED_FRAMES along time axis
    T = mfcc.shape[1]
    if T < EXPECTED_FRAMES:
        pad = np.zeros((NUM_MFCCS, EXPECTED_FRAMES - T), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=1)
    elif T > EXPECTED_FRAMES:
        mfcc = mfcc[:, :EXPECTED_FRAMES]

    # IMPORTANT: Do NOT transpose (you trained with (18, T))
    return mfcc.astype(np.float32)  # (18, 20)


def load_song(path):
    ad, sr = sf.read(path)
    song = ad[0:SAMPLE_RATE] #make sure to take 1s
    return song

def run_inference(mfcc_18xT):
    # Shape to (1, 18, T)
    x = mfcc_18xT[np.newaxis, :, :]

    # Quantize if input is INT8; else pass float
    if in_dtype == np.int8:
        if in_scale == 0.0:
            raise RuntimeError("INT8 model but missing input quantization params.")
        x_q = np.round(x / in_scale + in_zp).astype(np.int8)
        interpreter.set_tensor(i_details["index"], x_q)
    else:
        interpreter.set_tensor(i_details["index"], x)

    interpreter.invoke()

    y = interpreter.get_tensor(o_details["index"])[0]
    # Dequantize output if needed
    if out_dtype == np.int8 and out_scale > 0:
        y = (y.astype(np.int32) - out_zp) * out_scale
    y = y.astype(np.float32)

    if not np.isclose(np.sum(y), 1.0, atol=1e-2):
        e = np.exp(y - np.max(y))
        y = e / np.sum(e)

    idx = int(np.argmax(y))
    return idx, float(y[idx]), y

def classify_from_mic(mode):
    if not mode:
        y = record_seconds()
    else:
        filepath = os.path.join(HOME_DIR,"dataset", mode)
        y = load_song(filepath)
        
        pygame.mixer.init()
        sound = pygame.mixer.Sound(filepath)
        playing = sound.play()
        while playing.get_busy():
            pygame.time.delay(100)

    mfcc = extract_mfcc_like_training(y)
    idx, p, probs = run_inference(mfcc)

    print("\nTop-3 predictions:")
    for k in np.argsort(-probs)[:3]:
        print(f"{LIST_GENRES[k]:8s}: {probs[k]:.3f}")
    print(f"\nPredicted: {LIST_GENRES[idx]} ({p:.3f})")


if __name__ == "__main__":

    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode")
    mode = ap.parse_args().mode

    classify_from_mic(mode)
