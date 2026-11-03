# Functions with default arguments, f-strings and comprehensions.
def greet(name: str, times: int = 2) -> str:
    return ", ".join([f"hello {name}"] * times)


print(greet("wasm"))
print([n * n for n in range(10) if n % 2 == 0])
