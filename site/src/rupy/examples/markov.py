# A character-level Markov chain: learn which letter follows each pair, then babble.
class Rng:
    """A linear congruential generator (the constants of C's rand)."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state * 1103515245 + 12345) % 2147483648
        return self.state >> 8

    def below(self, n: int) -> int:
        return self.next() % n

text = """a neural network is a function made of layers. each layer multiplies by
weights and adds a bias. training adjusts the weights to make the loss small.
the gradient tells us which way the loss goes up, so we step the other way."""
text = " ".join(text.split())
table: dict[str, list[str]] = {}
for i in range(len(text) - 2):
    key = text[i : i + 2]
    table.setdefault(key, []).append(text[i + 2])
rng = Rng(7)
out = "th"
for _ in range(200):
    nxt = table.get(out[-2:], [" "])
    out += nxt[rng.below(len(nxt))]
print(out)
print(len(table), "two-letter states")
