from transformer_blocks import *

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass


@dataclass
class Configuration:
    block_size = 4
    vocab_size = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer = 8
    n_head = 16
    n_embd = 768
    dropout = 0.2
    bias = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class BigramLanguageModel(nn.Module):
    """
    A Transformer-based language model for generating text.

    Parameters:
        vocab_size (int): Size of the vocabulary.
        n_embd (int): Number of embedding dimensions.
        block_size (int): Maximum block size for input sequences.
        n_head (int): Number of attention heads.
        n_layer (int): Number of transformer layers.
        dropout (float): Dropout probability.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, n_head=config.n_head, block_size=config.block_size, dropout=config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    

    def forward(self, idx, targets=None):
        """
        Perform a forward pass through the BigramLanguageModel.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            targets (torch.Tensor): Target tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of logits for language model predictions.
            torch.Tensor or None: Loss tensor if targets are provided, otherwise None.
        """
        device = idx.device
        b, t = idx.size()
        if targets is not None:
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)/

        tok_emb = self.token_embedding_table(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.position_embedding_table(pos)  # position embeddings of shape (1, t, n_embd)

        x = tok_emb + pos_emb  # add token embeddings and position embeddings
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets, also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new text given a conditioning sequence.

        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Temperature parameter for sampling. Higher values make the output more diverse.
            top_k (int or None): Number of top-k options to consider for sampling. If None, all options are considered.

        Returns:
            torch.Tensor: Generated text tensor of shape (batch_size, sequence_length + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by the desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
