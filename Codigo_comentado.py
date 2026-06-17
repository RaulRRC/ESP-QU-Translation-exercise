"""
Transformer para Traducción Español → Quechua
==============================================
Implementación desde cero de un Transformer encoder-decoder (Vaswani et al., 2017)
entrenado sobre el dataset "spanish-to-quechua" de HuggingFace.

Soporta tres estrategias de tokenización: WordLevel, BPE y Unigram.
La evaluación se realiza con BLEU, chrF++ y COMET.

Uso:
    python Codigo.py --Tokenizer 0 --Train 1   # entrenar con WordLevel
    python Codigo.py --Tokenizer 1 --Train 0   # evaluar con BPE
"""

# ──────────────────────────────────────────────────────────────────────────────
# Librerías estándar
# ──────────────────────────────────────────────────────────────────────────────
import math
# import copy
import argparse        # Parseo de argumentos de línea de comandos
import time            # Medición del tiempo de entrenamiento por época
import random          # Semilla de aleatoriedad para reproducibilidad
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Procesamiento de datos y NLP
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import unicodedata                          # Normalización de texto Unicode
from datasets import Dataset as hf_Dataset # Dataset de HuggingFace (alias para claridad)
import sacrebleu                            # Métricas BLEU y chrF++
import os
import tokenizers                           # Tokenizadores BPE / WordLevel / Unigram (HuggingFace)

# ──────────────────────────────────────────────────────────────────────────────
# Ciencia de datos y visualización
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import seaborn as sns
from datasets import load_dataset

# ──────────────────────────────────────────────────────────────────────────────
# PyTorch
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# COMET – métrica neuronal de calidad de traducción
# ──────────────────────────────────────────────────────────────────────────────
from comet import download_model, load_from_checkpoint

class TranslationDataset(torch.utils.data.Dataset):
    """
    Dataset de traducción español → quechua compatible con PyTorch DataLoader.

    Cada elemento es un par (oración_español, oración_quechua) en texto plano.
    El texto quechua se envuelve con los tokens especiales [start] y [end]
    para el entrenamiento teacher-forced del decoder.
    """

    def __init__(self, text_pairs):
        # Lista de tuplas (es, que) con las oraciones en texto plano
        self.text_pairs = text_pairs
 
    def __len__(self):
        return len(self.text_pairs)
 
    def __getitem__(self, idx):
        es, que = self.text_pairs[idx]
        # El decoder recibe la secuencia objetivo envuelta con [start]/[end]
        return es, "[start] " + que + " [end]"
 
def test_tokenizer(es_tokenizer):
    """
    Prueba de sanidad del tokenizador español.

    Codifica una oración de ejemplo, muestra los tokens e IDs resultantes,
    y verifica que la decodificación recupere el texto original.
    """
    fr_sample = "Hola como estas ?"
    encoded = es_tokenizer.encode("[start] " + fr_sample + " [end]")
    print(f"Original: {fr_sample}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")

    print(f"Decoded: {es_tokenizer.decode(encoded.ids)}")
    
def collate_fn(batch):
    """
    Función de colación para el DataLoader: tokeniza y alinea los lotes.

    Convierte listas de cadenas de texto a tensores de IDs con padding
    uniforme dentro del lote. Usa los tokenizadores globales es_tokenizer
    y que_tokenizer para el español y el quechua respectivamente.

    Args:
        batch: lista de tuplas (str_español, str_quechua) del Dataset.

    Returns:
        Tupla de tensores (es_ids, que_ids) de forma (B, max_seq_len).
    """
    global es_tokenizer, que_tokenizer
    es_str, que_str = zip(*batch)
    # Codificación en lote; el padding se aplica automáticamente al máximo del lote
    es_enc = es_tokenizer.encode_batch(es_str, add_special_tokens=True)
    que_enc = que_tokenizer.encode_batch(que_str, add_special_tokens=True)
    es_ids = [enc.ids for enc in es_enc]
    que_ids = [enc.ids for enc in que_enc]
    return torch.tensor(es_ids), torch.tensor(que_ids)

def load_comet():
    """
    Descarga y carga el modelo COMET para evaluación de calidad de traducción.

    Utiliza el checkpoint 'wmt22-comet-da', un modelo de referencia estándar
    que evalúa la calidad de la traducción comparando hipótesis con referencias
    y fuentes. Requiere conexión a internet la primera vez.

    Returns:
        Modelo COMET listo para llamar a .predict().
    """
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    return comet_model

#All we need is attention ...
def scaled_dot_product_attention(
    Q: torch.Tensor,              # (batch, heads, seq_q, d_k)
    K: torch.Tensor,              # (batch, heads, seq_k, d_k)
    V: torch.Tensor,              # (batch, heads, seq_k, d_v)
    mask: Optional[torch.Tensor] = None,  # (batch, 1, seq_q, seq_k)
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vaswani et al. (2017) Equation (1):
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Returns:
        output  : (batch, heads, seq_q, d_v)
        weights : (batch, heads, seq_q, seq_k)  — for visualization
    """
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores  (batch, heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply optional mask (causal or padding)
    # mask=True  →  keep,  mask=False  →  fill with -inf  →  softmax≈0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax over key dimension → attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Replace NaN from all-masked rows (happens at padded positions)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Step 4: Weighted sum of values
    output = torch.matmul(attn_weights, V)
    return output, attn_weights




class MultiHeadAttention(nn.Module):
    """Vaswani et al. (2017) Equations (2–4)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads  # per-head dimension

        # Four projection matrices: Q, K, V projections + output projection
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # stored for visualization

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, heads, S, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        B, H, S, d_k = x.shape
        x = x.transpose(1, 2).contiguous()   # (B, S, H, d_k)
        return x.view(B, S, H * d_k)         # (B, S, d_model)

    def forward(
        self,
        query: torch.Tensor,        # (B, S_q, d_model)
        key:   torch.Tensor,        # (B, S_k, d_model)
        value: torch.Tensor,        # (B, S_k, d_model)
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project and split into heads
        Q = self.split_heads(self.W_q(query))  # (B, H, S_q, d_k)
        K = self.split_heads(self.W_k(key))    # (B, H, S_k, d_k)
        V = self.split_heads(self.W_v(value))  # (B, H, S_k, d_k)

        # Scaled dot-product attention for all heads in parallel
        attn_output, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )  # (B, H, S_q, d_k)

        # Concatenate heads and project
        combined = self.combine_heads(attn_output)  # (B, S_q, d_model)
        return self.W_o(combined)                   # (B, S_q, d_model)

class FeedForward(nn.Module):
    """
    Red feed-forward posicional (FFN). Vaswani et al. (2017) Ecuación (2).

    Aplica dos transformaciones lineales con activación GELU entre ellas:
        FFN(x) = Linear(GELU(Linear(x)))

    La dimensión interna d_ff suele ser 4× mayor que d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # Proyección de expansión
            nn.GELU(),                   # Activación no lineal (alternativa a ReLU del paper original)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),    # Proyección de contracción de vuelta a d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, S, d_model)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Vaswani et al. (2017) Equations (5–6)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the full encoding table once at construction time
        pe = torch.zeros(max_len, d_model)              # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)   # (max_len, 1)

        # Exponent term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

        # Register as buffer (not a parameter — not trained)
        self.register_buffer('pe', pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)  →  add PE for positions 0..S-1
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------- Visualize positional encodings -----------------------------------
def plot_positional_encoding(d_model: int = 64, max_len: int = 60):
    pe_layer = PositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_layer.pe[0].detach().numpy()   # (max_len, d_model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Heatmap
    sns.heatmap(pe, ax=axes[0], cmap='RdBu_r', center=0,
                cbar_kws={'label': 'PE value'})
    axes[0].set_title('Sinusoidal PE Heatmap\n(rows = positions, cols = dims)',
                       fontsize=11)
    axes[0].set_xlabel('Encoding dimension')
    axes[0].set_ylabel('Position')

    # Individual dimension curves
    colors = ['#534AB7', '#1D9E75', '#EF9F27', '#D85A30']
    for idx, (dim, c) in enumerate(zip([0, 4, 16, 32], colors)):
        axes[1].plot(pe[:, dim], label=f'dim {dim}', color=c, linewidth=1.8)
    axes[1].set_title('PE values across positions\n(selected dimensions)', fontsize=11)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('PE value')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    # plt.show()
    print('Lower dims oscillate FAST (local context).')
    print('Higher dims oscillate SLOW (global context).')
    plt.savefig("Positional_Encoding" )



def Test_attention():
    # ---------- Quick sanity check -------------------------------------------
    B, H, S, d_k = 2, 4, 6, 16
    Q_ = torch.randn(B, H, S, d_k)
    K_ = torch.randn(B, H, S, d_k)
    V_ = torch.randn(B, H, S, d_k)
    out_, weights_ = scaled_dot_product_attention(Q_, K_, V_)
    print(f'output shape   : {out_.shape}')      # (2, 4, 6, 16)
    print(f'weights shape  : {weights_.shape}')  # (2, 4, 6, 6)
    print(f'weights sum    : {weights_[0,0].sum(dim=-1)}')  # all 1.0


class EncoderLayer(nn.Module):
    """
    Single encoder layer:
        x → Self-Attention → Add&Norm → FFN → Add&Norm

    We use Pre-LN (LayerNorm before sublayer) for training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention with residual
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, src_mask))

        # Pre-LN FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer:
        x → Masked Self-Attention → Add&Norm
          → Cross-Attention (over encoder output) → Add&Norm
          → FFN → Add&Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = FeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,           # decoder input
        enc_out:  torch.Tensor,           # encoder output (K, V for cross-attn)
        src_mask: Optional[torch.Tensor] = None,  # padding mask
        tgt_mask: Optional[torch.Tensor] = None,  # causal mask
    ) -> torch.Tensor:
        # Masked self-attention (cannot attend to future positions)
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))

        # Cross-attention: decoder queries, encoder keys & values
        x_norm = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x_norm, enc_out, enc_out, src_mask))

        # FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    """
    Encoder del Transformer: apila N capas EncoderLayer.

    Pipeline por capa:
        tokens → Embedding (escalado) → PositionalEncoding → N × EncoderLayer → LayerNorm
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # LayerNorm final tras todas las capas
        # Escalar embeddings por sqrt(d_model) (Vaswani et al. Sección 3.4)
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding escalado + codificación posicional
        x = self.pos_enc(self.embedding(src) * self.scale)
        # Pasar por cada capa del encoder
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)   # (B, S_src, d_model)


class Decoder(nn.Module):
    """
    Decoder del Transformer: apila N capas DecoderLayer.

    Pipeline por capa:
        tokens → Embedding (escalado) → PositionalEncoding
              → N × DecoderLayer(enc_out, src_mask, tgt_mask) → LayerNorm
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt: torch.Tensor, enc_out: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pos_enc(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)   # (B, S_tgt, d_model)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:     int = 256,
        num_heads:   int = 8,
        d_ff:        int = 1024,
        num_layers:  int = 4,
        dropout:     float = 0.1,
        max_len:     int = 512,
    ):
        super().__init__()
        self.encoder    = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder    = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: share embedding weights with output projection
        # (Press & Wolf, 2017 — improves perplexity)
        if src_vocab_size == tgt_vocab_size:
            self.output_proj.weight = self.encoder.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for all linear/embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: position i can only attend to j <= i."""
        # Returns (1, 1, size, size) bool tensor
        mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
        return mask.bool()

    @staticmethod
    def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """(B, S) → (B, 1, 1, S) bool mask: True where token is NOT padding."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src: torch.Tensor,   # (B, S_src) token ids
        tgt: torch.Tensor,   # (B, S_tgt) token ids (teacher-forced)
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ) -> torch.Tensor:       # (B, S_tgt, tgt_vocab_size) logits
        src_mask = self.make_padding_mask(src, src_pad_idx)  # (B,1,1,S_src)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)  # (1,1,S,S)
        tgt_pad  = self.make_padding_mask(tgt, tgt_pad_idx)  # (B,1,1,S_tgt)
        tgt_mask = tgt_mask & tgt_pad  # combine causal + padding

        enc_out = self.encoder(src, src_mask)                # (B, S_src, d)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)  # (B, S_tgt, d)
        return self.output_proj(dec_out)                     # (B, S_tgt, V)

def test_model_summary():
    # ---------- Model summary ---------------------------------------------------
    SRC_VOCAB = TGT_VOCAB = 10000
    model = Transformer(SRC_VOCAB, TGT_VOCAB, d_model=256, num_heads=8,
                        d_ff=1024, num_layers=4).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters    : {total_params:,}')
    print(f'Trainable parameters: {trainable:,}')

    # Dummy forward pass
    src_dummy = torch.randint(1, SRC_VOCAB, (4, 20)).to(DEVICE)
    tgt_dummy = torch.randint(1, TGT_VOCAB, (4, 18)).to(DEVICE)
    logits = model(src_dummy, tgt_dummy)
    print(f'Output logits shape : {logits.shape}')  # (4, 18, 10000)

    # Demonstrate with a toy sentence using our scratch model
    sentence_tokens = ['The', 'animal', 'didn\'t', 'cross', 'the', 'street', 'because', 'it', 'was', 'tired']
    S = len(sentence_tokens)

    # Create dummy token ids
    toy_src = torch.randint(1, SRC_VOCAB, (1, S)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        # Extract attention weights from layer 0 of the encoder
        enc_embedding = model.encoder.pos_enc(
            model.encoder.embedding(toy_src) * model.encoder.scale
        )
        src_mask = model.make_padding_mask(toy_src)
        _ = model.encoder.layers[0](enc_embedding, src_mask)
        attn_w = model.encoder.layers[0].self_attn.attn_weights  # (1, H, S, S)

    attn_np = attn_w[0].cpu().numpy()  # (H, S, S)
    visualize_attention_heads(attn_np, sentence_tokens, sentence_tokens,
                            title='Encoder layer 1 — self-attention (untrained)')
    print('Note: weights are from an untrained model (random). Run Part 3 for meaningful patterns.')


class WarmupScheduler:
    """Vaswani et al. (2017) Equation (3)."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        t = self._step
        return (self.d_model ** -0.5) * min(t ** -0.5, t * self.warmup_steps ** -1.5)

def plot_scheduler():
    # Visualize the schedule
    d_model, warmup = 256, 1000
    steps = np.arange(1, 50001)
    lrs = (d_model ** -0.5) * np.minimum(steps ** -0.5, steps * warmup ** -1.5)

    plt.figure(figsize=(10, 3.5))
    plt.plot(steps, lrs, color='#534AB7', linewidth=2)
    plt.axvline(warmup, color='#1D9E75', linestyle='--', linewidth=1.5, label=f'warmup = {warmup} steps')
    plt.xlabel('Training step'); plt.ylabel('Learning rate'); plt.grid(alpha=0.3)
    plt.title(f'Vaswani LR schedule (d_model={d_model})', fontsize=12)
    plt.legend(); plt.tight_layout(); plt.show()
    print(f'Peak LR at step {warmup}: {lrs[warmup-1]:.6f}')
    plt.savefig("Scheduler")

def visualize_attention_heads(
    attention_matrix: np.ndarray,   # (num_heads, seq_q, seq_k)
    tokens_q: list,
    tokens_k: list,
    title: str = 'Attention weights',
):
    """
    Plot one heatmap per attention head.
    Brighter cells = higher attention weight.
    """
    num_heads = attention_matrix.shape[0]
    cols = 4
    rows = math.ceil(num_heads / cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()

    cmap = plt.cm.get_cmap('Purples')
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attention_matrix[h], cmap=cmap, aspect='auto',
                       vmin=0, vmax=attention_matrix.max())
        ax.set_xticks(range(len(tokens_k)))
        ax.set_yticks(range(len(tokens_q)))
        ax.set_xticklabels(tokens_k, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(tokens_q, fontsize=7)
        ax.set_title(f'Head {h+1}', fontsize=9)

    # Hide unused subplots
    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def Get_dataset():

    print("\nLoading dataset from HuggingFace…")
    ds = load_dataset("somosnlp-hackathon-2022/spanish-to-quechua")
    return ds

def dataset_filtrado(dataset):
    """
    Limpia y filtra el dataset eliminando pares de mala calidad.

    Pasos:
        1. Extrae las columnas 'es' y 'qu' del campo anidado 'translation'.
        2. Calcula el número de palabras y caracteres por oración.
        3. Elimina duplicados y pares con oraciones vacías.
        4. Descarta pares con más de 127 tokens (límite del modelo).

    Args:
        dataset: HuggingFace Dataset con columna 'translation'.

    Returns:
        HuggingFace Dataset con columnas ['es', 'qu'] ya filtradas.
    """
    df = pd.DataFrame(dataset)
    if "translation" in df.columns:
        # Desanidar el diccionario {'es': ..., 'qu': ...}
        df["es"] = df["translation"].apply(lambda x: x["es"])
        df["qu"] = df["translation"].apply(lambda x: x["qu"])    
    df["es_words"] = df["es"].str.split().str.len()
    df["qu_words"] = df["qu"].str.split().str.len()

    df["es_chars"] = df["es"].str.len()
    df["qu_chars"] = df["qu"].str.len()

    df[["es", "qu"]].isnull().sum()

    df.drop_duplicates(inplace=True)

    # Eliminar vacíos en español y quechua
    df = df.loc[df["es_words"] != 0]
    df = df.loc[df["qu_words"] != 0]

    # Eliminar pares muy largos (superan el MAX_LEN del Transformer)
    df = df.loc[df["es_words"] <= 127]
    df = df.loc[df["qu_words"] <= 127]

    return hf_Dataset.from_pandas(df[['es','qu']], preserve_index=False)

def normalize(line):
    """Normalize a line of text and split into two at the tab character"""
    line = unicodedata.normalize("NFKC", line.strip().lower())
    eng, fra = line.split("\t")
    return eng.lower().strip(), fra.lower().strip()

def Prepare_Dataset(ds):
    """
    Prepara los splits train / test / validation del dataset.

    Aplica dataset_filtrado a cada split y los convierte a listas de tuplas
    (oración_español, oración_quechua) listas para el DataLoader.

    Args:
        ds: Dataset de HuggingFace con splits 'train', 'test' y 'validation'.

    Returns:
        Tres listas de tuplas: text_pairs_train, text_pairs_test, text_pairs_validation.
    """
    train_data = ds["train"]
    train_data = dataset_filtrado(train_data)

    test_data   = ds["test"]
    test_data = dataset_filtrado(test_data)

    validation_data = ds["validation"]
    validation_data = dataset_filtrado(validation_data)

    # Convertir a lista de tuplas (es, qu)
    text_pairs_train = list(zip(train_data['es'], train_data['qu']))
    text_pairs_test = list(zip(test_data['es'], test_data['qu']))
    text_pairs_validation = list(zip(validation_data['es'], validation_data['qu']))

    # Mostrar las primeras 20 muestras de entrenamiento para verificación
    for i in text_pairs_train[0:20]: 
        print(i)
    return text_pairs_train, text_pairs_test, text_pairs_validation


# ──────────────────────────────────────────────────────────────────────────────
# Preparación de tokenizadores
# ──────────────────────────────────────────────────────────────────────────────

def Prepare_BPE(text_pairs_train):
    """
    Prepara tokenizadores Byte-Pair Encoding (BPE) para español y quechua.

    Si ya existen archivos guardados en ./BPE/, los carga directamente.
    En caso contrario, entrena nuevos tokenizadores BPE con ByteLevel
    pre-tokenizer (maneja correctamente caracteres especiales y Unicode)
    y los guarda para usos futuros.

    Args:
        text_pairs_train: lista de tuplas (es, qu) del conjunto de entrenamiento.

    Returns:
        Tupla (es_tokenizer, que_tokenizer).
    """
    if os.path.exists("./BPE/es_tokenizer.json") and os.path.exists("./BPE/que_tokenizer.json"):
        es_tokenizer = tokenizers.Tokenizer.from_file("./BPE/es_tokenizer.json")
        que_tokenizer = tokenizers.Tokenizer.from_file("./BPE/que_tokenizer.json")
    else:
        es_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[unk]"))
        que_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[unk]"))
    
        # ByteLevel pre-tokenizer: divide en bytes, añade prefijo de espacio
        es_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
        que_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
    
        # Decoder ByteLevel: elimina el símbolo de límite de palabra "Ġ" al decodificar
        es_tokenizer.decoder = tokenizers.decoders.ByteLevel()
        que_tokenizer.decoder = tokenizers.decoders.ByteLevel()
    
        # Entrenar BPE con vocabulario de 12 000 tokens
        VOCAB_SIZE = 12000
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=["[start]", "[end]", "[pad]","[unk]"],
            show_progress=True
        )
        es_tokenizer.train_from_iterator([x[0] for x in text_pairs_train], trainer=trainer)
        que_tokenizer.train_from_iterator([x[1] for x in text_pairs_train], trainer=trainer)
    
        # Activar padding automático al ID del token [pad]
        es_tokenizer.enable_padding(pad_id=es_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        que_tokenizer.enable_padding(pad_id=que_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
    
        # Persistir tokenizadores entrenados
        es_tokenizer.save("./BPE/es_tokenizer.json", pretty=True)
        que_tokenizer.save("./BPE/que_tokenizer.json", pretty=True)
    return es_tokenizer, que_tokenizer
    

def Prepare_WordLevel(text_pairs_train):
    """
    Prepara tokenizadores WordLevel (vocabulario de palabras completas).

    Si ya existen archivos guardados en ./Wordlevel/, los carga directamente.
    El pre-tokenizer usa espacios en blanco como delimitadores.
    Vocabulario máximo: 12 000 palabras; las palabras desconocidas se mapean a [unk].

    Args:
        text_pairs_train: lista de tuplas (es, qu) del conjunto de entrenamiento.

    Returns:
        Tupla (es_tokenizer, que_tokenizer).
    """
    if os.path.exists("./Wordlevel/es_tokenizer.json") and os.path.exists("./Wordlevel/que_tokenizer.json"):
        print("Cargando tokenizadores desde Google Drive...")
        es_tokenizer = tokenizers.Tokenizer.from_file("./Wordlevel/es_tokenizer.json")
        que_tokenizer = tokenizers.Tokenizer.from_file("./Wordlevel/que_tokenizer.json")

    else:
        es_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[unk]"))
        que_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token="[unk]"))
        # Separación por espacios en blanco
        es_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        que_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        VOCAB_SIZE = 12000

        trainer = tokenizers.trainers.WordLevelTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=[
                "[unk]",
                "[start]",
                "[end]",
                "[pad]"
            ],
            show_progress=True
        )

        es_tokenizer.train_from_iterator([x[0] for x in text_pairs_train], trainer=trainer)
        que_tokenizer.train_from_iterator([x[1] for x in text_pairs_train], trainer=trainer)
        es_tokenizer.enable_padding(pad_id=es_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        que_tokenizer.enable_padding(pad_id=que_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

        es_tokenizer.save("./Wordlevel/es_tokenizer.json", pretty=True)
        que_tokenizer.save("./Wordlevel/que_tokenizer.json", pretty=True)
    return es_tokenizer, que_tokenizer
 

def Prepare_Unigram(text_pairs_train):
    """
    Prepara tokenizadores Unigram Language Model para español y quechua.

    El modelo Unigram selecciona la segmentación que maximiza la probabilidad
    de la secuencia bajo un modelo de lenguaje de unigramas aprendido.
    Es especialmente útil para lenguas morfológicamente ricas como el quechua.

    Si ya existen archivos guardados en ./Unigram/, los carga directamente.

    Args:
        text_pairs_train: lista de tuplas (es, qu) del conjunto de entrenamiento.

    Returns:
        Tupla (es_tokenizer, que_tokenizer).
    """
    if os.path.exists("./Unigram/es_tokenizer.json") and os.path.exists("./Unigram/que_tokenizer.json"):
        es_tokenizer = tokenizers.Tokenizer.from_file("./Unigram/es_tokenizer.json")
        que_tokenizer = tokenizers.Tokenizer.from_file("./Unigram/que_tokenizer.json")
    else:
        # Crear tokenizadores Unigram
        es_tokenizer = tokenizers.Tokenizer(tokenizers.models.Unigram())
        que_tokenizer = tokenizers.Tokenizer(tokenizers.models.Unigram())

        # Pre-tokenizer ByteLevel (igual que BPE)
        es_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
        que_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Decoder ByteLevel
        es_tokenizer.decoder = tokenizers.decoders.ByteLevel()
        que_tokenizer.decoder = tokenizers.decoders.ByteLevel()

        VOCAB_SIZE = 12000

        # Entrenador Unigram
        trainer = tokenizers.trainers.UnigramTrainer(
            vocab_size=VOCAB_SIZE,
            unk_token="[unk]",
            special_tokens=["[start]", "[end]", "[pad]", "[unk]"],
            show_progress=True,
        )

        # Entrenar ambos tokenizadores
        es_tokenizer.train_from_iterator([x[0] for x in text_pairs_train], trainer=trainer)
        que_tokenizer.train_from_iterator([x[1] for x in text_pairs_train], trainer=trainer)

        # Activar padding
        es_tokenizer.enable_padding(pad_id=es_tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        que_tokenizer.enable_padding(pad_id=que_tokenizer.token_to_id("[pad]"), pad_token="[pad]")

        # Guardar tokenizadores
        es_tokenizer.save("./Unigram/es_tokenizer.json", pretty=True)
        que_tokenizer.save("./Unigram/que_tokenizer.json", pretty=True)
    return es_tokenizer, que_tokenizer
def save_evaluation_scores(
    csv_path,
    hypotheses,
    references,
    comet_model=None,
    data=None,
    experiment_name=None,
):
    """
    Compute BLEU, chrF++, COMET and save results to CSV.

    Parameters
    ----------
    csv_path : str
        Output CSV file.
    hypotheses : list[str]
        Model translations.
    references : list[str]
        Reference translations.
    comet_model : COMET model, optional
        Loaded COMET model.
    data : list[dict], optional
        COMET input format:
        [{"src": ..., "mt": ..., "ref": ...}, ...]
    experiment_name : str, optional
        Name of the experiment/run.
    """

    # BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    # chrF++
    chrfpp = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2
    )

    # COMET
    comet_score = None
    if comet_model is not None and data is not None:
        result = comet_model.predict(
            data,
            batch_size=8,
            gpus=1 if torch.cuda.is_available() else 0
        )
        comet_score = result.system_score

    row = {
        "experiment": experiment_name,
        "BLEU": bleu.score,
        "chrF++": chrfpp.score,
        "COMET": comet_score,
        "BLEU_1gram": bleu.precisions[0],
        "BLEU_2gram": bleu.precisions[1],
        "BLEU_3gram": bleu.precisions[2],
        "BLEU_4gram": bleu.precisions[3],
        "BP": bleu.bp,
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)

    return row



import torch
import sacrebleu
from tqdm import tqdm

# ─────────────────────────────────────────────
# Pre-cache all special token IDs once globally
# ─────────────────────────────────────────────
_START_ID = None
_END_ID   = None
_PAD_ES   = None
_PAD_QUE  = None

def _init_token_ids(es_tokenizer,que_tokenizer):
    global _START_ID, _END_ID, _PAD_ES, _PAD_QUE
    if _START_ID is None:
        _START_ID = que_tokenizer.token_to_id("[start]")
        _END_ID   = que_tokenizer.token_to_id("[end]")
        _PAD_ES   = es_tokenizer.token_to_id("[pad]")
        _PAD_QUE  = que_tokenizer.token_to_id("[pad]")


@torch.no_grad()
def beam_decode(model,es_tokenizer,que_tokenizer, es_ids, beam_size=4, max_len=60):
    """
    Optimized beam decoder.

    Key changes vs original:
    - Token IDs are cached after the first call (no dict lookup per call).
    - All active beams are stacked into a single tensor and decoded in ONE
      batched forward pass per timestep (beam_size × seq_len) instead of
      beam_size separate passes.
    - Candidate selection uses torch.topk on a flat GPU tensor instead of
      sorting a Python list, keeping work on the GPU.
    - Early exit when every beam has produced [end].
    """
    _init_token_ids(es_tokenizer,que_tokenizer)

    # Reject sequences that would exceed the positional embedding table
    if es_ids.shape[1] >= 128:
        return []

    device = es_ids.device

    # Encoder runs exactly once per source sentence
    src_mask = model.make_padding_mask(es_ids, _PAD_ES)
    enc_out  = model.encoder(es_ids, src_mask)          # (1, S, d_model)

    # ── initialise beams ──────────────────────────────────────────────────
    # Each beam: (cumulative_log_prob, [token_ids])
    beams: list[tuple[float, list[int]]] = [(0.0, [_START_ID])]
    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        # Separate finished beams from active ones
        active, newly_done = [], []
        for score, ids in beams:
            (newly_done if ids[-1] == _END_ID else active).append((score, ids))
        completed.extend(newly_done)

        if not active:
            break

        # ── batched decoder forward pass ───────────────────────────────
        # Stack all active beam sequences into (B, T) where B = len(active)
        max_t  = max(len(ids) for _, ids in active)
        tgt    = torch.full((len(active), max_t), _PAD_QUE,
                            dtype=torch.long, device=device)
        for i, (_, ids) in enumerate(active):
            tgt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        T       = tgt.shape[1]
        causal  = model.make_causal_mask(T, device)         # (T, T)
        tgt_pad = model.make_padding_mask(tgt, _PAD_QUE)    # (B, 1, 1, T)

        # Expand encoder output to match batch size of active beams
        enc_expanded = enc_out.expand(len(active), -1, -1)  # (B, S, d)
        src_mask_exp = src_mask.expand(len(active), -1, -1, -1)

        dec_out   = model.decoder(tgt, enc_expanded, src_mask_exp,
                                  causal & tgt_pad)          # (B, T, d)
        logits    = model.output_proj(dec_out[:, -1, :])     # (B, V)
        log_probs = torch.log_softmax(logits, dim=-1)        # (B, V)

        # ── top-k expansion (on GPU) ────────────────────────────────────
        topk_vals, topk_ids = log_probs.topk(beam_size, dim=-1)  # (B, K)

        candidates: list[tuple[float, list[int]]] = []
        for b, (score, ids) in enumerate(active):
            for k in range(beam_size):
                new_score = score + topk_vals[b, k].item()
                new_ids   = ids + [topk_ids[b, k].item()]
                candidates.append((new_score, new_ids))

        if not candidates:
            break

        # Length-normalised selection — sort once, keep top beam_size
        candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = candidates[:beam_size]

    completed.extend(beams)
    if not completed:
        return []

    completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
    return completed[0][1]


# ─────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────

EVAL_PAIRS = [
    (
        'Pero ¿qué piensa de este mundo, es decir, de la sociedad humana injusta alejada de Dios?',
        'Ichaqa, ¿imatam piensanki Diosmanta karunchasqa mana allin ruraq runakunamanta?'
    ),
    (
        '¿Cómo debe verse la paternidad, y por qué?',
        '¿Imaynatam qawana tayta - mama kayta, hinaspa imanasqa?'
    ),
    (
        '"Hablar aliviaba la angustia que sentía ", recuerda.',
        'Chaymi nin: "Willakusqaywanmi hawkayayta tarirqani ", nispa.'
    ),
    (
        'La revista Maclean\'s resume así la opinión de un conocido ateo: "Este concepto cristiano — la existencia de algo que ni la ciencia puede explicar ni nuestros sentidos percibir — [...] le resta valor a la única vida con que contamos y nos hace más propensos a la violencia ".',
        'Diospi mana iñiq reqsisqa runapa nisqanmantam huk revista nin: "Diospi iñiyqa [...] vidanchiktam yanqachan hinaspam maqanakuyman tanqawanchik ", nispa.'
    ),
    (
        'Anímelo a expresar su postura.',
        'Kallpanchay imayna piensasqanmanta willasunaykipaq.'
    ),
]

def run_evaluation(model = None,comet_model=None,es_tokenizer=None,que_tokenizer= None, text_pairs=None, experiment_name="Transformer_BPE_256"):
    """
    Evaluate translations with the optimised beam decoder.

    Args:
        text_pairs: iterable of (src, ref) pairs. Defaults to EVAL_PAIRS.
        experiment_name: label passed to save_evaluation_scores.
    """
    # if text_pairs is None:
    #     text_pairs = text_pairs_validation  # noqa: F821 – defined in calling scope

    hypotheses: list[str] = []
    references:  list[str] = []
    data:        list[dict] = []

    _init_token_ids(es_tokenizer,que_tokenizer)  # warm the cache before the loop

    print("Evaluating translations...")
    for src, ref in tqdm(text_pairs):
        es_ids = (
            torch.tensor(es_tokenizer.encode(src).ids)   # noqa: F821
                 .unsqueeze(0)
                 .to(DEVICE)                              # noqa: F821
        )

        token_ids = beam_decode(model,es_tokenizer,que_tokenizer, es_ids)            # noqa: F821
        hyp = que_tokenizer.decode(token_ids)             # noqa: F821

        data.append({"src": src, "mt": hyp, "ref": ref})
        hypotheses.append(hyp)
        references.append(ref)

    scores = save_evaluation_scores(                      # noqa: F821
        csv_path="translation_results_2.csv",
        hypotheses=hypotheses,
        references=references,
        comet_model=comet_model,                          # noqa: F821
        data=data,
        experiment_name=experiment_name,
    )
    print(scores)
    return scores

def load_model(dir, cfg, es_tokenizer, que_tokenizer):
    model = Transformer(
        src_vocab_size = len(es_tokenizer.get_vocab()),
        tgt_vocab_size = len(que_tokenizer.get_vocab()),
        d_model        = cfg.D_MODEL,
        num_heads      = cfg.NUM_HEADS,
        d_ff           = cfg.D_FF,
        num_layers     = cfg.NUM_LAYERS,
        dropout        = cfg.DROPOUT,
        max_len        = cfg.MAX_LEN,
    ).to(DEVICE)
    model.load_state_dict(torch.load(dir, weights_only=True)['model_state'])
    model.eval()
    return model


def main(Tokenizer, Train):
    """
    Punto de entrada principal: entrenamiento o evaluación del Transformer.

    Flujo de entrenamiento (Train=1):
        1. Cargar y filtrar el dataset.
        2. Preparar el tokenizador seleccionado.
        3. Construir DataLoaders.
        4. Instanciar el Transformer con los hiperparámetros de Config.
        5. Configurar pérdida (CrossEntropy + label smoothing) y optimizador Adam.
        6. Scheduler: warmup lineal → cosine annealing.
        7. Bucle de épocas: train → validate → (cada 50 épocas) BLEU → checkpoint.
        8. Guardar histórico de pérdidas en CSV.

    Flujo de evaluación (Train=0):
        1. Cargar tokenizador y modelo guardado.
        2. Calcular BLEU, chrF++ y COMET sobre el conjunto de validación.

    Args:
        Tokenizer: entero 0 (WordLevel), 1 (BPE) o 2 (Unigram).
        Train:     1 para entrenar, 0 para evaluar.
    """
    global es_tokenizer, que_tokenizer, optimizer, scheduler, loss_fn, cfg
    cfg = Config()
    ds = Get_dataset()
    text_pairs_train, text_pairs_test, text_pairs_validation = Prepare_Dataset(ds)

    # Seleccionar el tokenizador según el argumento recibido
    if Tokenizer == 1:
        es_tokenizer, que_tokenizer = Prepare_BPE(text_pairs_train)
        CHECKPOINT_PATH = "./BPE/checkpoint.pt"
    if Tokenizer == 2:
        es_tokenizer, que_tokenizer = Prepare_Unigram(text_pairs_train)
        CHECKPOINT_PATH = "./Unigram/checkpoint.pt"
    else:
        es_tokenizer, que_tokenizer = Prepare_WordLevel(text_pairs_train)
        CHECKPOINT_PATH = "./Wordlevel/checkpoint.pt"

    test_tokenizer(es_tokenizer)

    # Construir DataLoaders para cada split
    dataset_train = TranslationDataset(text_pairs_train)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    dataset_test = TranslationDataset(text_pairs_test)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    dataset_validation = TranslationDataset(text_pairs_validation)
    dataloader_validation = DataLoader(dataset_validation, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if Train == True:
        # ── Instanciar el modelo ─────────────────────────────────────────────
        model = Transformer(
            src_vocab_size = len(es_tokenizer.get_vocab()),
            tgt_vocab_size = len(que_tokenizer.get_vocab()),
            d_model        = cfg.D_MODEL,
            num_heads      = cfg.NUM_HEADS,
            d_ff           = cfg.D_FF,
            num_layers     = cfg.NUM_LAYERS,
            dropout        = cfg.DROPOUT,
            max_len        = cfg.MAX_LEN,
        ).to(DEVICE)

        # CrossEntropy con label smoothing; ignora el token [pad] en el cálculo
        loss_fn = nn.CrossEntropyLoss(ignore_index=que_tokenizer.token_to_id("[pad]"), label_smoothing=cfg.LABEL_SMOOTHING)
        # Optimizador Adam con hiperparámetros del paper original (Vaswani et al.)
        optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)

        # Scheduler: fase de warmup lineal seguida de decaimiento coseno
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_STEPS)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS * len(dataloader_train) - cfg.WARMUP_STEPS, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[cfg.WARMUP_STEPS])

        train_losses, val_losses, BLEU = [], [], []
        Best_val_loss = 100
        start_epoch = 0
        global_step = 0
        
        # Reanudar desde checkpoint si existe
        if os.path.exists(CHECKPOINT_PATH):
            print("Resuming from checkpoint...")
            start_epoch, global_step, last_loss = load_checkpoint(
                CHECKPOINT_PATH, model, optimizer, scheduler
            )

        # ── Bucle de entrenamiento ───────────────────────────────────────────
        for epoch in range(start_epoch, cfg.EPOCHS):
            bleu_score = 0
            global_step += 1
            start = time.time()
            train_loss, _ = train_one_epoch(model, dataloader_train, DEVICE)
            val_loss, _   = validate(model, dataloader_test, DEVICE)
            end = time.time()

            # Calcular BLEU cada 50 épocas (operación costosa con beam search)
            if epoch % 50 == 0 and epoch != 0:
                bleu_score, _ = Calculate_BLEU(model, text_pairs_validation[0:125], DEVICE)
                BLEU.append(bleu_score)
            
            train_losses.append(train_loss / len(dataloader_train))
            val_losses.append(val_loss / len(dataloader_test))
            
            # Guardar checkpoint solo si mejora la pérdida de validación
            if val_loss / len(dataloader_test) < Best_val_loss:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_loss, CHECKPOINT_PATH)
                torch.save({'epoch': epoch, 'model_state': model.state_dict()}, f"Best_model.pth")
                Best_val_loss = val_loss / len(dataloader_test)
            
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR {optimizer.param_groups[0]['lr']:.6f} | "
                f"Train loss {train_loss/len(dataloader_train):.4f} | "
                f"Val loss {val_loss/len(dataloader_test):.4f} | "
                f"Time {end-start:.1f}s")
            
        # Persistir el historial de pérdidas para análisis posterior
        save_losses_to_csv(train_losses, val_losses, "History.csv")

    else:
        # ── Modo evaluación ──────────────────────────────────────────────────
        comet_model = load_comet()
        if Tokenizer == 1:
            model = load_model("./BPE/Best_model.pth", cfg, es_tokenizer, que_tokenizer)
        if Tokenizer == 2:
            model = load_model("./Unigram/Best_model.pth", cfg, es_tokenizer, que_tokenizer)
        else:
            model = load_model("./WordLevel/Best_model.pt", cfg, es_tokenizer, que_tokenizer)

        run_evaluation(model, comet_model, es_tokenizer, que_tokenizer, text_pairs_validation, "Experiment")(model, loader, device):
    """
    Ejecuta una época completa de entrenamiento con teacher forcing.

    En teacher forcing, el decoder recibe la secuencia objetivo correcta
    en cada paso en lugar de su propia predicción anterior, lo que
    acelera y estabiliza el entrenamiento.

    La pérdida se calcula desplazando la secuencia objetivo un token:
        - Entrada del decoder: que_ids[:, :-1]  (todos menos el último)
        - Etiquetas objetivo:  que_ids[:, 1:]   (todos menos el [start])

    Args:
        model:  instancia del Transformer.
        loader: DataLoader de entrenamiento.
        device: dispositivo de cómputo (cpu / cuda).

    Returns:
        Tupla (epoch_loss_total, num_skips) donde num_skips es el número
        de lotes omitidos por exceder MAX_LEN.
    """
    global optimizer, scheduler, loss_fn, cfg
    model.train()
    skips = 0
    epoch_loss = 0 
    for es_ids, que_ids in loader:
        # Mover tensores al dispositivo de cómputo
        es_ids = es_ids.to(device)
        que_ids = que_ids.to(device)
        # Omitir secuencias que superen el límite posicional del modelo
        if es_ids.shape[1] >= 128:
            skips += 1
            continue
        # Paso hacia adelante
        optimizer.zero_grad()
        outputs = model(es_ids, que_ids)    
        # Pérdida: logits[t] predice token[t+1]; desplazamiento de 1 posición
        loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), que_ids[:, 1:].reshape(-1))
        loss.backward()
        # Gradient clipping para evitar explosión de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()                    
    return epoch_loss, skips


@torch.no_grad()
def validate(model, loader, device):
    """
    Calcula la pérdida de validación sin actualizar los pesos.

    Idéntico a train_one_epoch pero sin backward ni step del optimizador.
    Decorado con @torch.no_grad() para ahorrar memoria y acelerar el cómputo.

    Args:
        model:  instancia del Transformer en modo eval.
        loader: DataLoader de validación o test.
        device: dispositivo de cómputo.

    Returns:
        Tupla (val_loss_total, num_skips).
    """
    global optimizer, scheduler, loss_fn
    model.eval()
    skips = 0
    val_loss = 0
    for es_ids, que_ids in loader:
        es_ids = es_ids.to(device)
        que_ids = que_ids.to(device)
        if es_ids.shape[1] >= 128:
            skips += 1
            continue
        outputs = model(es_ids, que_ids)    
        loss = loss_fn(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), que_ids[:, 1:].reshape(-1))
        val_loss += loss.item()  
    return val_loss, skips    



def create_causal_mask(seq_len, device):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


@torch.no_grad()
def Calculate_BLEU(model, loader, device, beam_size=4, max_len=60):
    model.eval()
    skips = 0
    hypotheses = []
    references = []
    for es, ref in loader:
        es_ids = torch.tensor(es_tokenizer.encode(es).ids).unsqueeze(0).to(device)
        if es_ids.shape[1] >= 128:
            skips += 1
            continue        
        start_id = que_tokenizer.token_to_id("[start]")
        end_id   = que_tokenizer.token_to_id("[end]")
        pad_es   = es_tokenizer.token_to_id("[pad]")
        pad_que  = que_tokenizer.token_to_id("[pad]")
    
        src_mask = model.make_padding_mask(es_ids, pad_es)
        enc_out  = model.encoder(es_ids, src_mask)
        
        # Each beam: (score, token_ids)
        beams = [(0.0, [start_id])]
        completed = []
        
        

        for _ in range(max_len):
            candidates = []
            for score, ids in beams:
                if ids[-1] == end_id:
                    completed.append((score, ids))
                    continue
                tgt = torch.tensor([ids], device=es_ids.device)
                T = tgt.shape[1]
                causal  = model.make_causal_mask(T, es_ids.device)
                tgt_pad = model.make_padding_mask(tgt, pad_que)
                dec_out = model.decoder(tgt, enc_out, src_mask, causal & tgt_pad)
                logits  = model.output_proj(dec_out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                topk = log_probs.topk(beam_size)
                for lp, tok in zip(topk.values, topk.indices):
                    candidates.append((score + lp.item(), ids + [tok.item()]))
    
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = candidates[:beam_size]
    
        completed += beams
        completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        
        hyp = que_tokenizer.decode(completed[0][1])
        hypotheses.append(hyp)
        references.append(ref)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, skips

def create_padding_mask(batch, padding_token_id):
    batch_size, seq_len = batch.shape
    device = batch.device
    padded = torch.zeros_like(batch, device=device).float().masked_fill(batch == padding_token_id, float('-inf'))
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device) + padded[:,:,None] + padded[:,None,:]
    return mask[:, None, :, :]

class Config:
    """
    Hiperparámetros globales del modelo y del entrenamiento.

    Atributos del modelo:
        D_MODEL:    dimensión de los embeddings y de las capas internas.
        NUM_HEADS:  número de cabezas de atención (D_MODEL debe ser divisible por NUM_HEADS).
        D_FF:       dimensión interna de la red feed-forward.
        NUM_LAYERS: número de capas encoder y decoder.
        DROPOUT:    tasa de dropout aplicada en atención, FFN y embeddings.
        MAX_LEN:    longitud máxima de secuencia en tokens.

    Atributos de entrenamiento:
        EPOCHS:          número total de épocas.
        BATCH_SIZE:      tamaño de lote.
        WARMUP_STEPS:    pasos de warmup lineal del learning rate.
        LABEL_SMOOTHING: suavizado de etiquetas para regularización.
        CLIP_GRAD:       umbral de clipping de gradientes.

    Rutas:
        CHECKPOINT_DIR: directorio para guardar checkpoints.
        HISTORY_FILE:   JSON con el historial de pérdidas.
        PLOT_FILE:      gráfica del historial de entrenamiento.
    """
    # Arquitectura del modelo
    D_MODEL     = 256
    NUM_HEADS   = 4
    D_FF        = 1024
    NUM_LAYERS  = 4
    DROPOUT     = 0.2
    MAX_LEN     = 128       # Longitud máxima en tokens (debe coincidir con el límite en beam_decode)

    # Entrenamiento
    EPOCHS          = 400
    BATCH_SIZE      = 32
    WARMUP_STEPS    = 1000
    LABEL_SMOOTHING = 0.15
    CLIP_GRAD       = 1.0

    # Rutas de checkpointing y registro
    CHECKPOINT_DIR  = "checkpoints"
    HISTORY_FILE    = "training_history.json"
    PLOT_FILE       = "training_history.png"

def load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    """
    Restaura el estado completo de entrenamiento desde un archivo de checkpoint.

    Carga el modelo, optimizador, scheduler y (opcionalmente) el scaler de
    precisión mixta. El checkpoint almacena la época y el paso global para
    permitir reanudar el entrenamiento exactamente donde se dejó.

    Args:
        path:      ruta al archivo .pt del checkpoint.
        model:     instancia del Transformer (se actualiza in-place).
        optimizer: optimizador Adam (se actualiza in-place).
        scheduler: scheduler de learning rate (se actualiza in-place).
        scaler:    GradScaler para AMP (opcional).

    Returns:
        Tupla (start_epoch, start_step, loss) con los valores guardados.
    """
    checkpoint = torch.load(path, map_location="cpu")  # Cargar a CPU primero (más seguro)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1  # Reanudar desde la SIGUIENTE época
    start_step = checkpoint["step"]
    loss = checkpoint["loss"]

    return start_epoch, start_step, loss
import pandas as pd

def save_losses_to_csv(train_losses, val_losses, filename="losses.csv"):
    """
    Save train and validation losses to a CSV file.

    Parameters
    ----------
    train_losses : list
        List of training losses.
    val_losses : list
        List of validation losses.
    filename : str
        Output CSV filename.
    """
    max_len = max(len(train_losses), len(val_losses))

    # Pad shorter list with None
    train_losses = train_losses + [None] * (max_len - len(train_losses))
    val_losses = val_losses + [None] * (max_len - len(val_losses))

    df = pd.DataFrame({
        "epoch": range(1, max_len + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })

    df.to_csv(filename, index=False)
    print(f"Saved losses to {filename}")
    
def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path="checkpoint.pt"):
    """
    Guarda el estado completo de entrenamiento en un archivo .pt.

    Serializa el modelo, optimizador y scheduler junto con metadatos
    (época, paso global, pérdida) para poder reanudar el entrenamiento.
    Solo se guarda cuando la pérdida de validación mejora.

    Args:
        model:     instancia del Transformer entrenado.
        optimizer: estado del optimizador Adam.
        scheduler: estado del scheduler de learning rate.
        epoch:     época actual (0-indexada).
        step:      paso global actual.
        loss:      pérdida de entrenamiento de la época actual.
        path:      ruta destino del checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}, step {step}")


def load_model(dir, cfg, es_tokenizer, que_tokenizer):
    """
    Reconstruye e inicializa el Transformer desde un checkpoint de solo pesos.

    A diferencia de load_checkpoint, esta función solo carga los pesos del
    modelo (no el optimizador ni scheduler), lo que la hace adecuada para
    inferencia y evaluación.

    Args:
        dir:           ruta al archivo .pth con {'model_state': state_dict}.
        cfg:           instancia de Config con los hiperparámetros del modelo.
        es_tokenizer:  tokenizador del español (para obtener el tamaño del vocabulario).
        que_tokenizer: tokenizador del quechua.

    Returns:
        Transformer en modo eval() listo para inferencia.
    """
    model = Transformer(
        src_vocab_size = len(es_tokenizer.get_vocab()),
        tgt_vocab_size = len(que_tokenizer.get_vocab()),
        d_model        = cfg.D_MODEL,
        num_heads      = cfg.NUM_HEADS,
        d_ff           = cfg.D_FF,
        num_layers     = cfg.NUM_LAYERS,
        dropout        = cfg.DROPOUT,
        max_len        = cfg.MAX_LEN,
    ).to(DEVICE)
    model.load_state_dict(torch.load(dir, weights_only=True)['model_state'])
    model.eval()
    return model
    global es_tokenizer,que_tokenizer, optimizer, scheduler, loss_fn, cfg
    cfg = Config()
    ds = Get_dataset()
    text_pairs_train, text_pairs_test, text_pairs_validation = Prepare_Dataset(ds)
    if Tokenizer == 1:
        es_tokenizer, que_tokenizer = Prepare_BPE(text_pairs_train)
        CHECKPOINT_PATH = "./BPE/checkpoint.pt"
    if Tokenizer == 2:
        es_tokenizer, que_tokenizer = Prepare_Unigram(text_pairs_train)
        CHECKPOINT_PATH = "./Unigram/checkpoint.pt"
    else:
        es_tokenizer, que_tokenizer = Prepare_WordLevel(text_pairs_train)
        CHECKPOINT_PATH = "./Wordlevel/checkpoint.pt"

    test_tokenizer(es_tokenizer)

    dataset_train = TranslationDataset(text_pairs_train)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    dataset_test = TranslationDataset(text_pairs_test)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    dataset_validation = TranslationDataset(text_pairs_validation)
    dataloader_validation = DataLoader(dataset_validation, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if Train == True:
        model = Transformer(
            src_vocab_size = len(es_tokenizer.get_vocab()),
            tgt_vocab_size = len(que_tokenizer.get_vocab()),
            d_model        = cfg.D_MODEL,
            num_heads      = cfg.NUM_HEADS,
            d_ff           = cfg.D_FF,
            num_layers     = cfg.NUM_LAYERS,
            dropout        = cfg.DROPOUT,
            max_len        = cfg.MAX_LEN,
        ).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss(ignore_index=que_tokenizer.token_to_id("[pad]"),label_smoothing=cfg.LABEL_SMOOTHING)
        optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9,weight_decay=1e-5)


        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.WARMUP_STEPS)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS * len(dataloader_train) - cfg.WARMUP_STEPS, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[cfg.WARMUP_STEPS])
        train_losses, val_losses, BLEU = [], [], []

        Best_val_loss = 100
        start_epoch = 0
        global_step = 0
        
        # Resume if checkpoint exists
        if os.path.exists(CHECKPOINT_PATH):
            print("Resuming from checkpoint...")
            start_epoch, global_step, last_loss = load_checkpoint(
                CHECKPOINT_PATH, model, optimizer, scheduler
            )

        for epoch in range(start_epoch, cfg.EPOCHS):
            bleu_score = 0
            global_step += 1
            start = time.time()
            train_loss, _ = train_one_epoch(model,dataloader_train, DEVICE)
            val_loss, _ = validate(model, dataloader_test, DEVICE)
            end = time.time()

            if epoch % 50 == 0 and epoch != 0:
                bleu_score, _ = Calculate_BLEU(model, text_pairs_validation[0:125], DEVICE)
                BLEU.append(bleu_score)
            
            train_losses.append(train_loss/len(dataloader_train))
            val_losses.append(val_loss/len(dataloader_test))
            
            if val_loss/len(dataloader_test) < Best_val_loss:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_loss,CHECKPOINT_PATH)
                torch.save({'epoch': epoch, 'model_state': model.state_dict()},f"Best_model.pth")
                Best_val_loss = val_loss/len(dataloader_test)
            
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | LR {optimizer.param_groups[0]['lr']:.6f} | "
                f"Train loss {train_loss/len(dataloader_train):.4f} | "
                f"Val loss {val_loss/len(dataloader_test):.4f} | "
                f"Time {end-start:.1f}s")
            
        save_losses_to_csv(train_losses,val_losses, "History.csv" )

    else:
        comet_model = load_comet()
        if Tokenizer == 1:
            model = load_model("./BPE/Best_model.pth",cfg,es_tokenizer,que_tokenizer)
        if Tokenizer == 2:
            model = load_model("./Unigram/Best_model.pth",cfg,es_tokenizer,que_tokenizer)
        else:
            model = load_model("./WordLevel/Best_model.pt",cfg,es_tokenizer,que_tokenizer)

        run_evaluation(model, comet_model,es_tokenizer,que_tokenizer,text_pairs_validation,"Experiment")


if __name__ == "__main__":
    print("HOla mundo")

    # ── Argumentos de línea de comandos ─────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Tokenizer",
        type=int,
        choices=[0, 1, 2],
        required=True,
        help="Tokenizador a usar: 0 - WordLevel, 1 - BPE, 2 - Unigram",
        default='.'
    )
    parser.add_argument(
        "--Train",
        choices=[0, 1],
        type=int,
        required=True,
        help="Modo: 0 - Entrenar, 1 - Evaluar",
        default='.'
    )    
    args = parser.parse_args()

    # ── Reproducibilidad ─────────────────────────────────────────────────────
    # Fijar semillas en Python, NumPy y PyTorch para resultados deterministas
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Selección de dispositivo ─────────────────────────────────────────────
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    main(args.Tokenizer, args.Train)