import torch


class TextDataset:
    def __init__(self, tokens: torch.Tensor, block_size: int, batch_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size

    @classmethod
    def split(cls, tokens: torch.Tensor, block_size: int, batch_size: int, training_fraction: float = 0.9) -> tuple[TextDataset, TextDataset]:
        split = int(training_fraction * len(tokens))
        return cls(tokens[:split], block_size, batch_size), cls(tokens[split:], block_size, batch_size)

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(self.tokens) - self.block_size, (self.batch_size,))
        inputs = torch.stack([self.tokens[start : start + self.block_size] for start in starts])
        targets = torch.stack([self.tokens[start + 1 : start + self.block_size + 1] for start in starts])
        return inputs, targets
