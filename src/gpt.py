import torch
from torch import nn
from torch.nn import functional

from block import Block
from embedder import Embedder


class GPT(nn.Module):
    def __init__(self, vocabulary_size: int, block_size: int, embedding_size: int, number_of_blocks: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedder = Embedder(vocabulary_size, block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, block_size) for _ in range(number_of_blocks)])
        self.linear = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedder(tokens)
        embeddings = self.blocks(embeddings)
        return self.linear(embeddings)

    def calculate_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores = self(inputs)
        vocabulary_size = scores.shape[-1]
        return functional.cross_entropy(scores.view(-1, vocabulary_size), targets.view(-1))

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, number_of_new_tokens: int) -> torch.Tensor:
        for _ in range(number_of_new_tokens):
            context = tokens[:, -self.block_size :]
            scores = self(context)[:, -1, :]
            probabilities = torch.softmax(scores, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
