# Word counting with a dict, sorted by count.
text = """the quick brown fox jumps over the lazy dog
the dog barks and the fox runs"""
counts: dict[str, int] = {}
for w in text.split():
    counts[w] = counts.get(w, 0) + 1
top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
for word, n in top:
    print(f"{word:>6} {'#' * n} {n}")
print(len(counts), "distinct words;", sum(counts.values()), "in all")
