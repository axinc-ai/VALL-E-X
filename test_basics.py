from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import time

import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--onnx_export', action='store_true')
	parser.add_argument('--onnx_import', action='store_true')
	parser.add_argument('--benchmark', action='store_true')
	return parser.parse_args()

args = get_args()

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
音声合成のテストを行なっています。
"""
start = int(round(time.time() * 1000))
audio_array = generate_audio(text_prompt, args = args)
end = int(round(time.time() * 1000))
estimation_time = (end - start)
print(f'\t processing time {estimation_time} ms')

# save audio to disk
write_wav("vallex_generation.wav", SAMPLE_RATE, audio_array)
