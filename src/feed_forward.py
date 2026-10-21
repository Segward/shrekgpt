import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)
