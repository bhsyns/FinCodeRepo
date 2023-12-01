## Price of the call option in n periods
from scipy.special import binom
import numpy as np

def binomial_call(S, K, r, x, y, n=1):
    """
    Calculates the price of a call option using the binomial model.

    Parameters:
    - S: float, initial stock price
    - K: float, strike price
    - r: float, risk-free interest rate
    - x: float, upward movement of the stock price
    - y: float, downward movement of the stock price
    - n: int, number of periods

    Returns:
    - v0: float, price of the call option
    """
    u = (1 + x / 100)
    d = (1 - y / 100)

    # create the tree
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    for i in range(1, n + 1):
        tree[i, 0] = tree[i - 1, 0] * u
        for j in range(1, i + 1):
            tree[i, j] = tree[i - 1, j - 1] * d

    # compute the option price
    p = (1 + r - d) / (u - d)
    v0 = 0
    for j in range(n + 1):
        v0 += (max(tree[n, j] - K, 0) * (1 - p) ** j * p ** (n - j)) * binom(n, j)
    v0 = v0 / (1 + r) ** n

    return v0


def binomial_put(S, K, r, x, y, n=1):
    """
    Calculates the price of a put option using the binomial model.

    Parameters:
    - S: float, initial stock price
    - K: float, strike price
    - r: float, risk-free interest rate
    - x: float, upward movement of the stock price
    - y: float, downward movement of the stock price
    - n: int, number of periods

    Returns:
    - v0: float, price of the put option
    """
    u = (1 + x / 100)
    d = (1 - y / 100)

    # create the tree
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    for i in range(1, n + 1):
        tree[i, 0] = tree[i - 1, 0] * u
        for j in range(1, i + 1):
            tree[i, j] = tree[i - 1, j - 1] * d

    # compute the option price
    p = (1 + r - d) / (u - d)
    v0 = 0
    for j in range(n + 1):
        v0 += (max(K - tree[n, j], 0) * (1 - p) ** j * p ** (n - j)) * binom(n, j)
    v0 = v0 / (1 + r) ** n

    return v0


# American options

def binomial_call_american(S, K, r, x, y, n=1):
    """
    Calculates the price of an American call option using the binomial model.

    Parameters:
    - S: float, initial stock price
    - K: float, strike price
    - r: float, risk-free interest rate
    - x: float, upward movement of the stock price
    - y: float, downward movement of the stock price
    - n: int, number of periods

    Returns:
    - v0: float, price of the American call option
    """
    u = (1 + x / 100)
    d = (1 - y / 100)

    # create the tree
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    for i in range(1, n + 1):
        tree[i, 0] = tree[i - 1, 0] * u
        for j in range(1, i + 1):
            tree[i, j] = tree[i - 1, j - 1] * d

    # compute the option price
    p = (1 + r - d) / (u - d)
    v0 = 0
    for j in range(n + 1):
        v0 += (max(tree[n, j] - K, 0) * (1 - p) ** j * p ** (n - j)) * binom(n, j)
    v0 = v0 / (1 + r) ** n

    return v0


def binomial_put_american(S, K, r, x, y, n=1):
    """
    Calculates the price of an American put option using the binomial model.

    Parameters:
    - S: float, initial stock price
    - K: float, strike price
    - r: float, risk-free interest rate
    - x: float, upward movement of the stock price
    - y: float, downward movement of the stock price
    - n: int, number of periods

    Returns:
    - v0: float, price of the American put option
    """
    u = (1 + x / 100)
    d = (1 - y / 100)

    # create the tree
    tree = np.zeros((n + 1, n + 1))
    tree[0, 0] = S
    for i in range(1, n + 1):
        tree[i, 0] = tree[i - 1, 0] * u
        for j in range(1, i + 1):
            tree[i, j] = tree[i - 1, j - 1] * d

    option_tree = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        option_tree[n, j] = max(0, K - tree[n, j])

    # compute the american option price
    p = (1 + r - d) / (u - d)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            intrinsic = K - tree[i, j]
            option_tree[i, j] = max(intrinsic, (1 / (1 + r)) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1]))
    return option_tree[0, 0]
