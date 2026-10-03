"""Baseline: Erdos' construction. For a prime p, the points (x, x^2 mod p) have no
three on a line, giving p points on a p x p grid. Uses the largest prime p <= n."""


def _largest_prime_at_most(n):
    for p in range(n, 1, -1):
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            return p
    return 1


def construct(n, time_limit, seed):
    p = _largest_prime_at_most(n)
    return [(x, x * x % p) for x in range(p)]
