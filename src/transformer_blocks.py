import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    One head of self-attention.

    Parameters:
        head_size (int): The size of the attention head.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('upper_triangular_mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.upper_triangular_mask[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = torch.matmul(wei, v)  # (B, T, head_size)
        return out


class LayerNorm(nn.Module):
    """Layer normalization with an optional bias term."""

    def __init__(self, ndim, bias=True):
        """
        Initialize the LayerNorm module.

        Args:
            ndim (int): Number of features (dimensionality) in the input.
            bias (bool): Whether to include a bias term (default is True).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Apply LayerNorm to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Causal self-attention for the Transformer model."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize the CausalSelfAttention module.

        Args:
            n_embd (int): Number of embedding dimensions.
            n_head (int): Number of attention heads.
            block_size (int): Size of the causal mask.
        """
        super().__init__()
        assert n_embd % n_head == 0
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # Flash attention makes GPU faster, but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            if block_size < T:
                raise ValueError("block_size must be greater than or equal to T.")
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0))

    def forward(self, x):
        """
        Apply causal self-attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dimension).
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, and values for all heads in the batch and move the head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # Manual implementation of attention
            att = torch.bmm(q, k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = torch.bmm(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """A simple feedforward neural network with ReLU activation and dropout."""

    def __init__(self, n_embd, dropout):
        """
        Initialize the FeedForward module.

        Args:
            n_embd (int): Number of embedding dimensions.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(inplace=True),  # Use inplace=True for a small memory optimization
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Apply the feedforward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dimension).
        """
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initialize the Block module.

        Args:
            n_embd (int): Embedding dimension.
            n_head (int): Number of attention heads.
            block_size (int): Size of the causal mask.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Apply one block of the Transformer to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dimension).
        """
        # Apply self-attention with layer normalization
        x = x + self.sa(self.ln1(x))
        # Apply feedforward network with layer normalization
        x = x + self.ffwd(self.ln2(x))
        return x
