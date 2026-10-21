import torch

from dataset import TextDataset
from gpt import GPT


class Trainer:
    def __init__(self, model: GPT, training_dataset: TextDataset, testing_dataset: TextDataset, learning_rate: float) -> None:
        self.model = model
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train(self, number_of_steps: int, evaluation_interval: int) -> None:
        for step in range(number_of_steps + 1):
            if step % evaluation_interval == 0:
                training_loss = self.evaluate(self.training_dataset)
                testing_loss = self.evaluate(self.testing_dataset)
                print(f"Step {step}: training loss {training_loss:.3f}, testing loss {testing_loss:.3f}")

            inputs, targets = self.training_dataset.get_batch()
            loss = self.model.calculate_loss(inputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, dataset: TextDataset, number_of_batches: int = 20) -> float:
        self.model.eval()
        losses = [self.model.calculate_loss(*dataset.get_batch()).item() for _ in range(number_of_batches)]
        self.model.train()
        return sum(losses) / len(losses)
