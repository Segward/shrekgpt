import math

import torch
from torch import nn


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, block_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.shape[1]
        embedding_size = embeddings.shape[2]
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)

        scores = query @ key.transpose(-2, -1) / math.sqrt(embedding_size)
        scores = scores.masked_fill(self.mask[:sequence_length, :sequence_length] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ value
