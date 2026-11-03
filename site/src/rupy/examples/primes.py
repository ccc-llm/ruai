# The sieve of Eratosthenes up to 200,000.
def primes(n: int) -> list[int]:
    sieve = [True] * (n + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]


ps = primes(200000)
print(len(ps), ps[:10], ps[-3:])
