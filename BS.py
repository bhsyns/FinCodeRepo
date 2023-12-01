from scipy.integrate import quad
import numpy as np

def call(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the price of a call option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Price of the call option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    dm = dp - (vol * np.sqrt(T - t))

    Np = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dp)[0]
    Nn = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dm)[0]

    return S * np.exp(-D * (T - t)) * Np - K * np.exp(-r * (T - t)) * Nn


def put(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the price of a put option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Price of the put option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    dm = dp - (vol * np.sqrt(T - t))

    Np = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, -dp)[0]
    Nn = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, -dm)[0]

    return -S * np.exp(-D * (T - t)) * Np + K * np.exp(-r * (T - t)) * Nn


def power_call(T, K, S, vol, r, t=0, D=0, n=1):
    """
    Calculate the price of a power call option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.
    - n (float, optional): Power of the option. Default is 1.

    Returns:
    - float: Price of the power call option.
    """
    d1 = 1 / (vol * np.sqrt(T - t)) * (np.log(S / K ** (1 / n)) + (r - D + (n - 1 / 2) * vol ** 2) * (T - t))
    d2 = d1 - n * vol * np.sqrt(T - t)
    Nd1 = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, d1)[0]
    Nd2 = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, d2)[0]

    return S ** n * np.exp(((n - 1) * (r + n * vol ** 2 / 2) - n * D) * (T - t)) * Nd1 - K * np.exp(-r * (T - t)) * Nd2


def power_put(T, K, S, vol, r, t=0, D=0, n=1):
    """
    Calculate the price of a power put option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.
    - n (float, optional): Power of the option. Default is 1.

    Returns:
    - float: Price of the power put option.
    """
    d1 = 1 / (vol * np.sqrt(T - t)) * (np.log(S / K ** (1 / n)) + (r - D + (n - 1 / 2) * vol ** 2) * (T - t))
    d2 = d1 - n * vol * np.sqrt(T - t)

    Nd1 = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, -d1)[0]
    Nd2 = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, -d2)[0]

    return -S ** n * np.exp(((n - 1) * (r + n * vol ** 2 / 2) - n * D) * (T - t)) * Nd1 + K * np.exp(-r * (T - t)) * Nd2


def parity(c, p, S, K, r, T, t=0):
    """
    Verify the put-call parity relationship between a call and a put option.

    Parameters:
    - c (float): Price of the call option.
    - p (float): Price of the put option.
    - S (float): Current price of the underlying asset.
    - K (float): Strike price of the options.
    - r (float): Risk-free interest rate.
    - T (float): Time to expiration in years.
    - t (float, optional): Time to valuation in years. Default is 0.
    """
    print(c - p == S - K * np.exp(-r * (T - t)))


def delta_call(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the delta of a call option.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Delta of the call option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    Np = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dp)[0]

    return np.exp(-D * (T - t)) * Np


def delta_put(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the delta of a put option.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Delta of the put option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    Np = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dp)[0]

    return np.exp(-D * (T - t)) * (Np - 1)


def gamma_call_put(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the gamma of a call or put option.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Gamma of the option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    Npp = np.exp(-dp * dp / 2) / np.sqrt(2 * np.pi)

    return (np.exp(-D * (T - t)) * Npp) / (vol * S * np.sqrt(T - t))


def vega_call_put(T, K, St, vol, r, t=0, D=0):
    """
    Calculate the vega of a call or put option.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - St (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Vega of the option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(St * np.exp((r - D) * (T - t)) / K) + 1 / 2 * (vol * np.sqrt(T - t))
    return St * np.sqrt(T - t) * np.exp(-D * (T - t)) * np.exp(-dp * dp / 2) / np.sqrt(2 * np.pi)



def binary_call(T, S0, vol, r, t=0, D=0):
    """
    Calculate the price of a binary call option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - S0 (float): Initial price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Price of the binary call option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S0 * np.exp((r - D) * (T - t))) + 1 / 2 * (vol * np.sqrt(T - t))
    dm = dp - (vol * np.sqrt(T - t))
    Nn = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dm)[0]

    return np.exp(-r * (T - t)) * Nn


def binary_put(T, S0, vol, r, t=0, D=0):
    """
    Calculate the price of a binary put option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - S0 (float): Initial price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Price of the binary put option.
    """
    dn = 1 / (vol * np.sqrt(T - t)) * np.log(np.exp((r - D) * (T - t))) - 1 / 2 * (vol * np.sqrt(T - t))
    Nn = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dn)[0]

    return np.exp(-t * (T - t)) * (1 - Nn)


def digital_call(T, K, S, vol, r, t=0, D=0):
    """
    Calculate the price of a digital call option using the Black-Scholes formula.

    Parameters:
    - T (float): Time to expiration in years.
    - K (float): Strike price of the option.
    - S (float): Current price of the underlying asset.
    - vol (float): Volatility of the underlying asset.
    - r (float): Risk-free interest rate.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Price of the digital call option.
    """
    dp = 1 / (vol * np.sqrt(T - t)) * np.log(S * np.exp((r - D) * (T - t)) / K) - 1 / 2 * (vol * np.sqrt(T - t))
    Np = quad(lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi), -np.inf, dp)[0]

    return np.exp(-r * (T - t)) * Np
