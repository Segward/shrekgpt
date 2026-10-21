class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        characters = sorted(set(text))
        self.vocabulary_size = len(characters)
        self.character_to_token = {character: token for token, character in enumerate(characters)}
        self.token_to_character = {token: character for token, character in enumerate(characters)}

    def encode(self, text: str) -> list[int]:
        return [self.character_to_token[character] for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.token_to_character[token] for token in tokens)
