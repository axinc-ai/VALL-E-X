# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Dict, Iterator, List, Tuple, Union

import ailia
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from icefall.utils import make_pad_mask
# from torchmetrics.classification import MulticlassAccuracy

from data.input_strategies import PromptedFeatures
from modules.embedding import SinePositionalEmbedding, TokenEmbedding
from modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS
from .visualizer import visualize


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


# NOTE: There are two ways to implement the model
#       1) [VALL-F] standard TransformerDecoder, use x as memory
#       2) [VALL-E] modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
#          use x as the prefix of decoder inputs
class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = True,
        num_quantizers: int = 8,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        nar_d_model = int(d_model * nar_scale_factor)

        self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)

        # ID NUM_AUDIO_TOKENS     -> PAD
        # ID NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = TokenEmbedding(
            d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
        )

        # PreNet
        if add_prenet:
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            print("ar is identity")
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(
            d_model, NUM_AUDIO_TOKENS + 1, bias=False
        )

        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(num_quantizers - 1)
                ]
            )  # W_a

            # PreNet
            if add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_d_model, nar_d_model),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_d_model),
                )
            else:
                print("nar is identity")
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

            self.nar_text_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.0,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(
                    nar_d_model, norm=nn.LayerNorm(nar_d_model)
                )
                if norm_first
                else None,
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_d_model, 1)
                    for i in range(num_quantizers - 1)
                ]
            )

            if share_embedding:
                # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
                # NOTE(Feiteng): In the experiment, this undermines accuracy
                # self.ar_predict_layer.weight = self.ar_audio_embedding.weight

                # We also share the parameters of the acoustic embedding layer and the output prediction layer,
                # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[
                        j
                    ].weight = self.nar_audio_embeddings[j + 2].weight

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    print(f" AR parameter: {name}")
                    yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    print(f"NAR parameter: {name}")
                    yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # inputs, targets
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1),
                targets,
            )

        return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes, prefix_mode):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(0, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    codes[:, :prefix_len, j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](
                        codes[:, prefix_len:, j]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif prefix_mode in [2, 4]:
            if prefix_mode == 2:
                # random prefix
                prefix_len = min(225, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[
                        b, start : start + prefix_len, nar_stage
                    ] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    y_prompts_codes[..., j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        raise NotImplementedError

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: Union[torch.Tensor, None] = None,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError

    def visualize(
        self,
        predicts: Tuple[torch.Tensor],
        batch: Dict[str, Union[List, torch.Tensor]],
        output_dir: str,
        limit: int = 4,
    ) -> None:
        raise NotImplementedError


class VALLE(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(VALLE, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            **kwargs,
        )
        self.language_ID = {
            'en': 0,
            'zh': 1,
            'ja': 2,
        }
        self.ar_language_embedding = TokenEmbedding(d_model, len(self.language_ID))
        self.nar_language_embedding = TokenEmbedding(d_model, len(self.language_ID))

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ):
        raise NotImplementedError

    def audio_embedding(self, y):
        y_emb = self.ar_audio_embedding(y)
        y_emb = self.ar_audio_prenet(y_emb)
        y_pos = self.ar_audio_position(y_emb)
        return y_pos

    def export_decoder(self, xy_pos, xy_attn_mask, kv_cache, offset):
        xy_dec, kv_cache = self.ar_decoder.infer(
            xy_pos,
            mask=xy_attn_mask,
            past_kv=kv_cache,
            offset=offset
        )
        logits = self.ar_predict_layer(xy_dec[:, -1])
        return logits, kv_cache

    def export_nar_decoder(self, xy_pos, i):
        weights = torch.zeros((7, 1, 1024))
        for j in range(7):
            weights[j, :, :] = self.nar_stage_embeddings[j].weight # (1, 1024)
        weight = weights[i]
        xy_dec, _ = self.nar_decoder(
            (xy_pos, weight)
        )
        return xy_dec

    def export_predict_layers(self, xy_pos, i):
        results = torch.zeros((7, xy_pos.shape[0], xy_pos.shape[1], xy_pos.shape[2]))
        for j in range(7):
            results[j, :, :, :] = self.nar_predict_layers[j](xy_pos)
        return results[i].reshape((xy_pos.shape[0], xy_pos.shape[1], xy_pos.shape[2]))

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
        prompt_language: str = None,
        text_language: str = None,
        onnx_export = False,
        onnx_import = False,
        benchmark = False
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        # Add language embedding
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += self.ar_language_embedding(prompt_language_id)
        x[:, enroll_x_lens:, :] += self.ar_language_embedding(text_language_id)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        max_len = 1024 # TBD
        kv_cache = torch.zeros((12, 2, 1, 16, max_len, 64))
        kv_cache_numpy = np.zeros((12, 2, 1, 16, max_len, 64))
        offset = 0
        # torch.Size([1, 16, n, 64])が12レイヤー * 2ノード分ある

        use_kv_caching = True
        while True:
            if not onnx_import:
                y_pos = self.audio_embedding(y)
            if onnx_export:
                # kv_cacheの固定化が必要
                #print(y.shape) #(1, 24)
                #print(y_pos.shape) #(1,23,1024)

                if offset == 0:
                    print("Export audio_embedding to onnx")
                    self.forward = self.audio_embedding
                    torch.onnx.export(
                        self,
                        (y),
                        "audio_embedding.onnx",
                        input_names=["y"],
                        output_names=["y_pos"],
                        dynamic_axes={
                            "y": [1],
                            "y_pos": [1]
                        },
                        verbose=False, opset_version=15
                    )           
            if onnx_import:
                if offset == 0:
                    print("Impot audio_embedding from onnx")
                    anet = ailia.Net(weight="audio_embedding.onnx", env_id = 1, memory_mode = 11)
                start = int(round(time.time() * 1000))
                y_pos = anet.run([y.numpy()])[0]
                end = int(round(time.time() * 1000))
                y_pos = torch.from_numpy(y_pos)
                if benchmark:
                    print(f'ailia processing time {end - start} ms')

            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)


            if use_kv_caching and offset>=1:#kv_cache is not None:
                xy_pos = xy_pos[:, [-1]] # 前回のトークンは1つ
            else:
                pass # initial prompt

            if not onnx_import:
                start = int(round(time.time() * 1000))
                xy_dec, kv_cache = self.ar_decoder.infer(
                    xy_pos,
                    mask=xy_attn_mask,
                    past_kv=kv_cache,
                    offset=offset
                )
                logits = self.ar_predict_layer(xy_dec[:, -1])
                end = int(round(time.time() * 1000))
                if benchmark:
                    print(f'torch processing time {end - start} ms')

            if onnx_export:
                # kv_cacheの固定化が必要
                #print("ar_decoder input xy_pos", xy_pos.shape) # torch.Size([1, 1, 1024])
                #print("ar_decoder input mask", xy_attn_mask.shape) # torch.Size([61, 61])
                #print("ar_decoder input kv_cache", kv_cache.shape) # torch.Size([12, 2, 1, 16, 1024, 64])
                #print("ar_decoder output xy_dec", xy_dec.shape) # torch.Size([1, 1, 1024])
                #print("ar_decoder output kv_cache", kv_cache.shape) # torch.Size([12, 2, 1, 16, 1024, 64])

                if offset == 0:
                    print("Export ar_decoder to onnx")
                    self.forward = self.export_decoder
                    torch.onnx.export(
                        self,
                        (xy_pos, xy_attn_mask, kv_cache, offset),
                        "ar_decoder.onnx",
                        input_names=["xy_pos", "mask", "past_kv", "offset"],
                        output_names=["logits", "kv_cache"],
                        dynamic_axes={
                            "xy_pos": [1],
                            "mask": [0, 1],
                            "past_kv": [4],
                            "logits": [1],
                            "kv_cache": [4],
                        },
                        verbose=False, opset_version=15
                    )
            
            if onnx_import:
                if offset == 0:
                    print("Impot ar_decoder from onnx")
                    net = ailia.Net(weight="ar_decoder.onnx", env_id = 1, memory_mode = 11)
                offset_tensor = np.zeros((1))
                offset_tensor[0] = offset
                start = int(round(time.time() * 1000))
                logits, kv_cache_numpy = net.run([xy_pos.numpy(), xy_attn_mask.numpy(), kv_cache_numpy, offset_tensor])
                end = int(round(time.time() * 1000))
                logits = torch.from_numpy(logits)
                if benchmark:
                    print(f'ailia processing time {end - start} ms')

            offset = offset + xy_pos.shape[-2]

            samples = topk_sampling(
                logits, top_k=top_k, top_p=1, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        # Add language embedding
        prompt_language_id = torch.LongTensor(np.array([self.language_ID[prompt_language]])).to(x.device)
        if isinstance(text_language, str):
            text_language_id = torch.LongTensor(np.array([self.language_ID[text_language]])).to(x.device)
        elif isinstance(text_language, List):
            text_language_id = torch.LongTensor(np.array([self.language_ID[tl] for tl in text_language])).to(x.device)
        x[:, :enroll_x_lens, :] += self.nar_language_embedding(prompt_language_id)
        x[:, enroll_x_lens:, :] += self.nar_language_embedding(text_language_id)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                print("prefix_mode==0 ", i)

                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                print("prefix_mode!=0 ", i)

                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                if not onnx_import or onnx_export:
                    xy_dec, _ = self.nar_decoder(
                        (xy_pos, self.nar_stage_embeddings[i].weight)
                    )
                    logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                if onnx_export:
                    if i == 0:
                        # site-packages/torch/nn/functional.py:5346:0のviewを書き換えないとreshapeのshapeが固定されてしまう
                        # https://github.com/axinc-ai/ailia-models/issues/1236

                        print("Export nar_decoder to onnx")
                        print("xy_pos.shape",xy_pos.shape)
                        print("self.nar_weights.shape",self.nar_stage_embeddings[i].weight.shape)
                        self.forward = self.export_nar_decoder
                        iv = torch.tensor([i], dtype=torch.int64)
                        from torch.autograd import Variable
                        iv = Variable(iv)
                        torch.onnx.export(
                            self,
                            (xy_pos, iv),
                            "nar_decoder.onnx",
                            input_names=["xy_pos", "i"],
                            output_names=["xy_dec"],
                            dynamic_axes={
                                "xy_pos": {1:"num_sequence"},
                                "xy_dec": {1:"num_sequence"}
                            },
                            verbose=False, opset_version=15
                        )

                        print("Export nar_predict_layers to onnx")
                        self.forward = self.export_predict_layers
                        print("xy_dec.shape",xy_dec.shape)
                        torch.onnx.export(
                            self,
                            (xy_dec[:, text_len + prefix_len :], iv),
                            "nar_predict_layers.onnx",
                            input_names=["xy_pos", "i"],
                            output_names=["logits"],
                            dynamic_axes={
                                "xy_pos": {1:"num_sequence"},
                                "logits": {1:"num_sequence"}
                            },
                            verbose=False, opset_version=15
                        )           
                if onnx_import:
                    print("Impot nar_decoder from onnx")
                    if i == 0:
                        nar_decoder = ailia.Net(weight="nar_decoder.onnx", env_id = 1, memory_mode = 11)
                    offset_tensor = np.zeros((1))
                    offset_tensor[0] = i
                    print(xy_pos.shape, offset_tensor.shape)
                    xy_dec = nar_decoder.run([xy_pos.numpy(), offset_tensor])[0] # Normal inference
                    #xy_dec = nar_decoder.run([xy_pos.numpy()[:, 0:-1, :], offset_tensor])[0] # Test trim
                    end = int(round(time.time() * 1000))
                    xy_dec = torch.from_numpy(xy_dec)
                    if benchmark:
                        print(f'ailia processing time {end - start} ms')

                    print("Impot nar_predict_layers from onnx")
                    if i == 0:
                        nar_predict = ailia.Net(weight="nar_predict_layers.onnx", env_id = 1, memory_mode = 11)
                    logits = nar_predict.run([xy_dec[:, text_len + prefix_len :].numpy(), offset_tensor])[0]
                    end = int(round(time.time() * 1000))
                    logits = torch.from_numpy(logits)
                    if benchmark:
                        print(f'ailia processing time {end - start} ms')

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        # AR Decoder
        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        # Non-AR Decoders
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token
