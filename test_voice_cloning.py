import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--onnx_export', action='store_true')
	parser.add_argument('--onnx_import', action='store_true')
	parser.add_argument('--benchmark', action='store_true')
	return parser.parse_args()

args = get_args()

import time

model_name="jsut"

# generate audio embedding
from utils.prompt_making import make_prompt
start = int(round(time.time() * 1000))
#make_prompt(name=model_name, audio_prompt_path="VOICEACTRESS100_001.wav")
#make_prompt(name=model_name, audio_prompt_path="BASIC5000_0001.wav")
make_prompt(name=model_name, audio_prompt_path="BASIC5000_0001.wav", transcript="水をマレーシアから買わなくてはならないのです") # Disable whisper
#make_prompt(name=model_name, audio_prompt_path="kyakuno.wav")
end = int(round(time.time() * 1000))
estimation_time = (end - start)
print(f'\t processing time {estimation_time} ms')

# generate audio
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

preload_models()

start = int(round(time.time() * 1000))
text_prompt = """
音声合成のテストを行なっています。
"""
audio_array = generate_audio(text_prompt, prompt=model_name, args = args)
end = int(round(time.time() * 1000))
estimation_time = (end - start)
print(f'\t processing time {estimation_time} ms')

write_wav(model_name+"_cloned.wav", SAMPLE_RATE, audio_array)