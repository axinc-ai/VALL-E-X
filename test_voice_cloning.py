# generate audio embedding
from utils.prompt_making import make_prompt
model_name="jsut"
#make_prompt(name=model_name, audio_prompt_path="VOICEACTRESS100_001.wav")
make_prompt(name=model_name, audio_prompt_path="BASIC5000_0001.wav")
#make_prompt(name=model_name, audio_prompt_path="kyakuno.wav")

# generate audio
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

preload_models()

text_prompt = """
音声合成のテストを行なっています。
"""
audio_array = generate_audio(text_prompt, prompt=model_name)

write_wav(model_name+"_cloned.wav", SAMPLE_RATE, audio_array)