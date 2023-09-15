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

import math

import torch
import torch.nn as nn

import ailia
import numpy as np

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

        self.onnx_mode = False
        self.onnx_path = False

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        if self.onnx_mode:
            return self.forward_onnx(x)
        X = self.word_embeddings(x)
        X = self.dropout(X)
        return X

    def export_to_onnx(self, path):
        print("Export token embeddings to "+path)
        num_sequence = 10
        x = torch.zeros((1, num_sequence), dtype=torch.int64)
        torch.onnx.export(
            self,
            (x),
            path,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={
                "x": [1],
                "y": [1]
            },
            verbose=False, opset_version=15
        )
        self.onnx_mode = True
        self.onnx_path = path
    
    def forward_onnx(self, x: torch.Tensor):
        #print("Import token embeddings from "+self.onnx_path)
        anet = ailia.Net(weight=self.onnx_path, env_id = 1, memory_mode = 11)
        y = anet.run([x.numpy()])[0]
        y = torch.from_numpy(y)
        return y


class TokenEmbeddingLayers(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.ModuleList(
            nn.Embedding(self.vocab_size, self.dim_model)
            for i in range(7)
        )

        self.onnx_mode = False
        self.onnx_path = False
    
    def set_weight(self, emb : TokenEmbedding, layer_id):
        self.word_embeddings[layer_id].weight = emb.weight#word_embeddings.weight
        
    def forward(self, x: torch.Tensor, layer_id):
        if self.onnx_mode:
            return self.forward_onnx(x, layer_id)
       
        #print("Input", x.shape)

        results = torch.zeros((7, x.shape[0], x.shape[1], self.dim_model))
        for j in range(7):
            results[j, :, :, :] = self.word_embeddings[j](x)
        X = results[layer_id].reshape((x.shape[0], x.shape[1], self.dim_model))

        X = self.dropout(X)

        return X

    def import_from_onnx(self, path):
        self.onnx_mode = True
        self.onnx_path = path

    def export_to_onnx(self, path):
        #print("Export token embeddings to "+path)
        num_sequence = 10
        x = torch.zeros((1, num_sequence), dtype=torch.int64)
        layer_id = 0
        #layer_id = torch.tensor([layer_id], dtype=torch.int64)
        torch.onnx.export(
            self,
            (x, layer_id),
            path,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={
                "x": [1],
                "y": [1]
            },
            verbose=False, opset_version=15
        )

        self.onnx_mode = True
        self.onnx_path = path
    
    def forward_onnx(self, x: torch.Tensor, layer_id):
        #print("Import token embeddings from "+self.onnx_path)
        anet = ailia.Net(weight=self.onnx_path, env_id = 1, memory_mode = 11)
        layer_id = np.array(layer_id, dtype=np.int64)
        y = anet.run([x.numpy(), layer_id])[0]
        y = torch.from_numpy(y)
        return y


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        #self.pe = None
        #self.extend_pe(torch.tensor(0.0).expand(1, 4000))

        self.onnx_mode = False
        self.onnx_path = False

    def extend_pe(self, x):
        """Reset the positional encodings."""
        #if self.pe is not None:
        #    if self.pe.size(1) >= x.size(1):
        #        if self.pe.dtype != x.dtype or self.pe.device != x.device:
        #            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
        #        return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.to(device=x.device, dtype=x.dtype).detach()
        return pe

    def infer(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        pe = self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + alpha * pe[:, : x.size(1)]
        return self.dropout(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        alpha_tensor = torch.tensor(self.alpha[0])
        if self.onnx_mode:
            #print("input", x.shape)
            y = self.forward_onnx(x, alpha_tensor)
            #print("output", y.shape)
            return y
        return self.infer(x, alpha_tensor)

    def export_alpha(self, label, path):
        print(path, "Export position embedding alpha", self.alpha)

        self.onnx_mode = True
        self.onnx_path = path

    def export_onnx(self, path):
        backup = self.forward
        self.forward = self.infer

        print("Export position embeddings to "+path)
        x = torch.zeros((1, 46, self.dim_model)) # torch.Size([1, 46, 1024])
        torch.onnx.export(
            self,
            (x, self.alpha),
            path,
            input_names=["x", "alpha"],
            output_names=["y"],
            dynamic_axes={
                "x": [1],
                "y": [1]
            },
            verbose=False, opset_version=15
        )

        self.onnx_mode = True
        self.onnx_path = path

        self.forward = backup

    def forward_onnx(self, x: torch.Tensor, alpha: torch.Tensor):
        #print("Import token embeddings from "+self.onnx_path)
        anet = ailia.Net(weight=self.onnx_path, env_id = 1, memory_mode = 11)
        y = anet.run([x.numpy(), alpha.numpy()])[0]
        y = torch.from_numpy(y)
        return y