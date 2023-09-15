# coding: utf-8
import os
import torch
from vocos import Vocos
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import ailia
import time

import pathlib
import platform
if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from utils.sentence_cutter import split_text_into_sentences

from macros import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

url = 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'

checkpoints_dir = "./checkpoints/"

model_checkpoint_name = "vallex-checkpoint.pt"

model = None

codec = None

vocos = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

def preload_models():
    global model, codec, vocos
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
    if not os.path.exists(os.path.join(checkpoints_dir, model_checkpoint_name)):
        import wget
        try:
            logging.info(
                "Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
            # download from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt to ./checkpoints/vallex-checkpoint.pt
            wget.download("https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
                          out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_adaptive)
        except Exception as e:
            logging.info(e)
            raise Exception(
                "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
                "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))
    # VALL-E
    model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
    checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.eval()

    # Encodec
    codec = AudioTokenizer(device)
    
    vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

def export_vocos_head(x): # for onnx
    #print("linear weight", vocos.head.out.weight.shape) # torch.Size([1282, 384])
    x = vocos.head.out(x).transpose(1, 2)
    mag, p = x.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
    # wrapping happens here. These two lines produce real and imaginary value
    x = torch.cos(p)
    y = torch.sin(p)
    # recalculating phase here does not produce anything new
    # only costs time
    # phase = torch.atan2(y, x)
    # S = mag * torch.exp(phase * 1j)
    # better directly produce the complex value 
    return mag, x, y

def export_vocos_istft(x, y): # for onnx
    S = (x + 1j * y)
    n_fft = vocos.head.istft.n_fft # 1280
    hop_length = vocos.head.istft.hop_length # 320
    win_length = vocos.head.istft.win_length # 1280
    window = vocos.head.istft.window
    print("istft settings", n_fft, hop_length, win_length, window)
    audio = torch.istft(S, n_fft, hop_length, win_length, window, center=True)
    return audio

def export_vocos(frames):
    features = vocos.codes_to_features(frames)
    x = vocos.backbone(features)
    mag, x, y = export_vocos_head(x)
    return mag * x, mag * y


@torch.no_grad()
def generate_audio(text, prompt=None, language='auto', accent='no-accent', args=None):
    onnx_export = False
    onnx_import = False
    benchmark = False

    if args != None:
        onnx_export = args.onnx_export
        onnx_import = args.onnx_import
        benchmark = args.benchmark

    global model, codec, vocos, text_tokenizer, text_collater
    if args.onnx_export:
        model.export_token_embedding()
    if args.onnx_import:
        model.import_token_embedding()
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt is not None:
        prompt_path = prompt
        if not os.path.exists(prompt_path):
            prompt_path = "./presets/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            prompt_path = "./customs/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt_path)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = lang if lang != 'mix' else 'en'

    enroll_x_lens = text_prompts.shape[-1]
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    # accent control
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        onnx_export=onnx_export,
        onnx_import=onnx_import,
        benchmark=benchmark
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)

    if not onnx_import or onnx_export:
        features = vocos.codes_to_features(frames)
    
        # original
        #samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        # divide head
        x = vocos.backbone(features, bandwidth_id=torch.tensor([2], device=device))
        samples = vocos.head(x) # isftf

    if onnx_export:
        #print("vocos.codes_to_features input", frames.shape) # torch.Size([8, 1, 350])
        #print("vocos.codes_to_features output", features.shape) # torch.Size([1, 128, 350])

        #print("vocos.backbone input", features.shape) # torch.Size([1, 128, 350])
        #print("vocos.backbone output", x.shape) # torch.Size([1, 350, 384])

        #print("vocos.head input", x.shape) # torch.Size([1, 350, 384])
        #print("vocos.head output", samples.shape) # torch.Size([1, 112000])

        print("Export vocos to onnx")
        vocos.forward = export_vocos
        torch.onnx.export(
            vocos,
            (frames),
            "vocos.onnx",
            input_names=["frames"],
            output_names=["x", "y"],
            dynamic_axes={
                "frames": [2],
                "x": [2],
                "y": [2],
            },
            verbose=False, opset_version=15
        )           

    if onnx_import:
        print("Impot vocos from onnx")
        vnet = ailia.Net(weight="vocos.onnx", env_id = 1, memory_mode = 11)
        start = int(round(time.time() * 1000))
        x, y = vnet.run([frames.numpy()])
        end = int(round(time.time() * 1000))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if benchmark:
            print(f'ailia processing time {end - start} ms')
        samples = export_vocos_istft(x, y)

    return samples.squeeze().cpu().numpy()

@torch.no_grad()
def generate_audio_from_long_text(text, prompt=None, language='auto', accent='no-accent', mode='sliding-window'):
    """
    For long audio generation, two modes are available.
    fixed-prompt: This mode will keep using the same prompt the user has provided, and generate audio sentence by sentence.
    sliding-window: This mode will use the last sentence as the prompt for the next sentence, but has some concern on speaker maintenance.
    """
    global model, codec, vocos, text_tokenizer, text_collater
    if prompt is None or prompt == "":
        mode = 'sliding-window'  # If no prompt is given, use sliding-window mode
    sentences = split_text_into_sentences(text)
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]

    # if initial prompt is given, encode it
    if prompt is not None and prompt != "":
        prompt_path = prompt
        if not os.path.exists(prompt_path):
            prompt_path = "./presets/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            prompt_path = "./customs/" + prompt + ".npz"
        if not os.path.exists(prompt_path):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt_path)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = language if language != 'mix' else 'en'
    if mode == 'fixed-prompt':
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        for text in sentences:
            text = text.replace("\n", "").strip(" ")
            if text == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text = lang_token + text + lang_token

            enroll_x_lens = text_prompts.shape[-1]
            logging.info(f"synthesize text: {text}")
            phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
            text_tokens, text_tokens_lens = text_collater(
                [
                    phone_tokens
                ]
            )
            text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
            text_tokens_lens += enroll_x_lens
            # accent control
            lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=-100,
                temperature=1,
                prompt_language=lang_pr,
                text_language=langs if accent == "no-accent" else lang,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
        # Decode with Vocos
        frames = complete_tokens.permute(1,0,2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))
        return samples.squeeze().cpu().numpy()
    elif mode == "sliding-window":
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        original_audio_prompts = audio_prompts
        original_text_prompts = text_prompts
        for text in sentences:
            text = text.replace("\n", "").strip(" ")
            if text == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text = lang_token + text + lang_token

            enroll_x_lens = text_prompts.shape[-1]
            logging.info(f"synthesize text: {text}")
            phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
            text_tokens, text_tokens_lens = text_collater(
                [
                    phone_tokens
                ]
            )
            text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
            text_tokens_lens += enroll_x_lens
            # accent control
            lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=-100,
                temperature=1,
                prompt_language=lang_pr,
                text_language=langs if accent == "no-accent" else lang,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
            if torch.rand(1) < 0.5:
                audio_prompts = encoded_frames[:, :, -NUM_QUANTIZERS:]
                text_prompts = text_tokens[:, enroll_x_lens:]
            else:
                audio_prompts = original_audio_prompts
                text_prompts = original_text_prompts
        # Decode with Vocos
        frames = complete_tokens.permute(1,0,2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))
        return samples.squeeze().cpu().numpy()
    else:
        raise ValueError(f"No such mode {mode}")