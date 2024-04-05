import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import torch_geometric
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, BertGenerationConfig, BertGenerationDecoder
from fairseq.data import FastaDataset, EncodedFastaDataset, Dictionary, BaseWrapperDataset
import esm
import esm.inverse_folding
from omegaconf import DictConfig, OmegaConf
import hydra
from GPUtil import showUtilization as gpu_usage
from typing import List, Dict
import pandas as pd
from pandas import DataFrame as df
import time
import os
import json
import wandb
import csv
import random
from rfdiffusion.kinematics import xyz_to_t2d
from rfdiffusion.util import get_torsions, make_frame, torsion_indices, torsion_can_flip, reference_angles
from rfdiffusion.inference.utils import Denoise, process_target
from rfdiffusion.chemical import aa2long, aa2longalt, torsions, ideal_coords
from rfdiffusion.inference.model_runners  import Sampler
from rfdiffusion.diffusion import Diffuser
import diffusor_utils
class Pair2Pair(nn.module):
    #performs alphafold-style pairwise attention on 2d data
    def __init__(self, emb: int, num_heads: int):
        super(Pair2Pair, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.attention = nn.multiheadattention(emb, num_heads)
    def forward(self, pair:torch.tensor, pair_mask:torch.tensor) -> torch.tensor:
        #in: pair: (B, L, L, emb)
        #out: (B, L, L, emb) - updated with pairwise attention
        #performs columnwise self-attention on every index of the pair
        #then updates the pair with the attention scores
        B, L, _, _ = pair.shape
        returner = torch.zeros(B, L, L, self.emb)
        for i in range(L):
            attention_scores = self.attention(pair[:, i, :, :], pair[:, i, :, :], pair[:, i, :, :], attn_mask=pair_mask.repeat(self.num_heads, 1, 1))
            returner[:, i, :, :] = attention_scores
        return returner
class All2Lat(nn.module):
    #performs attention from 1d, 2d, and 3d tracks to latent space query
    def __init__(self, emb: int, num_heads: int):
        super(All2Lat, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.attention = nn.multiheadattention(emb, num_heads)
    def forward(self, latent: torch.tensor, kv:torch.tensor, seq_mask:torch.tensor) -> torch.tensor:
        #in: latent: (B, w, l, emb), kv: either (B, L, emb), (B, L, L, emb)
        #out: (B, w, l, emb) - updated with attention from kv
        #performs attention from kv to latent space
        #then updates the latent space with the attention scores
        if len(kv.shape) == 3:
            B, w, l, _ = latent.shape
            kv_flat = kv.view(B, -1, self.emb)
            return self.attention(latent.view(B, -1, self.emb), kv_flat, kv_flat, attn_mask=seq_mask.unsqueeze(1).repeat(self.num_heads, w*l, 1)).view(B, w, l, self.emb) 
        elif len(kv.shape) == 4:
            B, w, l, _ = latent.shape
            L = kv.shape[1]
            kv_flat = kv.view(B, -1, self.emb)
            return self.attention(latent.view(B, -1, self.emb), kv_flat, kv_flat, attn_mask=seq_mask.unsqueeze(1).repeat(self.num_heads, w*l, L)).view(B, w, l, self.emb)
        else:
            raise ValueError("kv must have shape (B, L, emb), (B, L, L, emb)")
class Lat2Seq(nn.module):
    def __init__(self, emb: int, num_heads: int):
        super(Lat2Seq, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.attention = nn.multiheadattention(emb, num_heads)
    def forward(self, seq: torch.tensor, latent:torch.tensor, seq_mask:torch.tensor) -> torch.tensor:
        #in: seq: (B, L, emb), latent: (B, w, l, emb)
        #out: (B, L, emb) - updated with attention from latent
        #performs attention from latent to sequence
        #then updates the sequence with the attention scores
        B, L, _ = seq.shape
        _, w, l, _ = latent.shape
        #attn_mask size is (num_heads, L, w*l), as target sequence is sequences, size (B, L, emb). source sequence is flattened latent space of size (B, w*l, emb)
        return self.attention(seq, latent.view(B, -1, self.emb), latent.view(B, -1, self.emb), attn_mask=seq_mask.repeat(self.num_heads, 1, w*l))
class Lat2Pair(nn.module):
    def __init__(self, emb: int, num_heads: int):
        super(Lat2Pair, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.attention = nn.multiheadattention(emb, num_heads)
    def forward(self, pair: torch.tensor, latent:torch.tensor, seq_mask:torch.tensor) -> torch.tensor:
        #in: pair: (B, L, L, emb), latent: (B, w, l, emb), seq_mask: (B, L)
        #out: (B, L, L, emb) - updated with attention from latent
        #performs attention from latent to pair
        #then updates the pair with the attention scores
        B, L, _, _ = pair.shape
        _, w, l, _ = latent.shape
        returner = torch.zeros(B, L, L, self.emb)

        for i in range(L):
            #attn_mask size is (num_heads, L, w*l), as target sequence is row of pair, size (B, L, emb). source sequence is flattened latent space of size (B, w*l, emb)
            attention_scores = self.attention(pair[:, i, :, :], latent.view(B, -1, self.emb), latent.view(B, -1, self.emb), attn_mask=seq_mask.unsqueeze(-1).repeat(self.num_heads, 1, w*l))
            returner[:, i, :, :] = attention_scores
        return returner
class B2Lat(nn.module):
    def __init__(self, emb: int, num_heads: int):
        super(B2Lat, self).__init__() 
        self.emb = emb
        self.num_heads = num_heads
        self.seq_attention = All2Lat(emb, num_heads)
        self.pair_attention = All2Lat(emb, num_heads)

    def forward(self, latent: torch.tensor, seq:torch.tensor, pair: torch.tensor, seq_mask:torch.tensor) -> torch.tensor:
        #in: latent: (B, w, l, emb), seq: (B, L, emb), pair: (B, L, L, emb)
        #out: (B, w, l, emb) - updated with attention from seq and pair
        #performs attention from seq and pair to latent space
        #then updates the latent space with the attention scores
        r = self.pair_attention(latent, pair, seq_mask)
        return self.seq_attention(r, seq, seq_mask)
        
class Lat2B(nn.module):
    def __init__(self, emb: int, num_heads: int):
        super(Lat2B, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.seq_attention = Lat2Seq(emb, num_heads)
        self.pair_attention = Lat2Pair(emb, num_heads)
    def forward(self, seq: torch.tensor, pair: torch.tensor, latent: torch.tensor, seq_mask: torch.tensor, pair_mask=torch.tensor) -> torch.tensor:
        #in: seq: (B, L, emb), pair: (B, L, L, emb), latent: (B, w, l, emb)
        #out: (B, L, emb), (B, L, L, emb) - updated with attention from latent
        #performs attention from latent to seq and pair
        #then updates the seq and pair with the attention scores
        seq = self.seq_attention(seq, latent, seq_mask)
        pair = self.pair_attention(pair, latent, seq_mask)
        return seq, pair
class Exchange(nn.module):
    def __init__(self, emb: int, num_heads: int):
        super(Exchange, self).__init__()
        self.emb = emb
        self.num_heads = num_heads
        self.lat2b = Lat2B(emb, num_heads)
        self.b2lat = B2Lat(emb, num_heads)
        self.pair2pair = Pair2Pair(emb, num_heads)
        self.seq2seq = nn.multiheadattention(emb, num_heads)
    def forward(self, seq: torch.tensor, pair: torch.tensor, latent: torch.tensor, seq_mask:torch.tensor, pair_mask:torch.tensor) -> torch.tensor:
        #in: seq: (B, L, emb), pair: (B, L, L, emb), latent: (B, w, l, emb), seq_mask: (B, L), pair_mask: (B, L, L)
        #out: (B, L, emb), (B, L, L, emb), (B, w, l, emb) - updated with attention from each other
        #performs attention from seq, pair, and latent to each other
        #then updates each with the attention scores
        seq, pair = self.lat2b(seq, pair, latent, seq_mask, pair_mask)
        pair = self.pair2pair(pair, pair_mask)
        seq = self.seq2seq(seq, seq, seq, attn_mask=pair_mask.repeat(self.num_heads, 1, 1))
        latent = self.b2lat(latent, seq, pair, seq_mask)
        return seq, pair, latent


class Monomer(nn.module):
    def __init__(
        num_comb_blocks,
        num_heads,
        embed_dim,
        
    ):
        super(Monomer, self).__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.modulelist([B2Lat(embed_dim, num_heads)], [Exchange(embed_dim, num_heads) for _ in range(num_comb_blocks)])
    def forward(self, seq: torch.tensor, pair: torch.tensor, latent: torch.tensor, seq_mask:torch.tensor, pair_mask:torch.tensor) -> torch.tensor:
        #in: seq: (B, L, emb), pair: (B, L, L, emb), latent: (B, w, l, emb), seq_mask: (B, L), pair_mask: (B, L, L)
        #out: (B, w, l, emb) - updated with attention from seq and pair
        #performs attention from seq and pair to latent space
        #then updates the latent space with the attention scores
        latent = self.get_latent()
        for block in self.blocks:
            seq, pair, latent = block(seq, pair, latent, seq_mask, pair_mask)
        return self.latent
    def get_latent(self, B, w_hidden, l_hidden, embed_dim=self.embed_dim, param=False):
        if param:
            return nn.parameter(torch.tensor((w_hidden, l_hidden, embed_dim))).unsqueeze(0).repeat(B, -1, -1, -1)
        else:
            return torch.tensor((w_hidden, l_hidden, embed_dim)).unsqueeze(0).repeat(B, -1, -1, -1)
        

class EncodeDecode(pl.LightningModule):

    