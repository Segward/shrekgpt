import torch
from torch import nn

from feed_forward import FeedForward
from single_head_attention import SingleHeadAttention


class Block(nn.Module):
    def __init__(self, embedding_size: int, block_size: int) -> None:
        super().__init__()
        self.attention = SingleHeadAttention(embedding_size, block_size)
        self.attention_normalization = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForward(embedding_size)
        self.feed_forward_normalization = nn.LayerNorm(embedding_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.attention_normalization(embeddings + self.attention(embeddings))
        embeddings = self.feed_forward_normalization(embeddings + self.feed_forward(embeddings))
        return embeddings
