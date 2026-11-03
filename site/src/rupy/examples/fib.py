# Fibonacci three ways: recursion with a memo, a loop, and big integers.
def fib(n: int, memo: dict[int, int]) -> int:
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]


memo: dict[int, int] = {}

print([fib(i, memo) for i in range(15)])
a, b = 0, 1
for _ in range(90):
    a, b = b, a + b
print("fib(90) =", a)
print("golden ratio ≈", fib(40, memo) / fib(39, memo))
