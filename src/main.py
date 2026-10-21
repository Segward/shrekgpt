from pathlib import Path

import torch

from dataset import TextDataset
from gpt import GPT
from tokenizer import CharacterTokenizer
from trainer import Trainer

DATA_PATH = Path(__file__).parent.parent / "data" / "shrek.txt"
BLOCK_SIZE = 64
EMBEDDING_SIZE = 64
NUMBER_OF_BLOCKS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUMBER_OF_TRAINING_STEPS = 3000
EVALUATION_INTERVAL = 300

def main():
    text = DATA_PATH.read_text()
    tokenizer = CharacterTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text))
    training_dataset, testing_dataset = TextDataset.split(tokens, BLOCK_SIZE, BATCH_SIZE)

    model = GPT(tokenizer.vocabulary_size, BLOCK_SIZE, EMBEDDING_SIZE, NUMBER_OF_BLOCKS)
    trainer = Trainer(model, training_dataset, testing_dataset, LEARNING_RATE)
    trainer.train(NUMBER_OF_TRAINING_STEPS, EVALUATION_INTERVAL)

    prompt = torch.tensor([tokenizer.encode("Shrek")])
    generated = model.generate(prompt, number_of_new_tokens=300)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
