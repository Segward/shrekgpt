import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, vocabulary_size: int, block_size: int, embedding_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_length = tokens.shape[1]
        positions = torch.arange(sequence_length, device=tokens.device)
        token_embeddings = self.token_embedding(tokens)
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings
