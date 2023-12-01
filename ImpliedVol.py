from scipy.integrate import quad
import numpy as np
from scipy.optimize import fsolve

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


def implied_vol(S0, K, T, r, c, t=0, D=0):
    """
    Calculate the implied volatility of an option using the Black-Scholes formula.

    Parameters:
    - S0 (float): Current price of the underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - c (float): Price of the option.
    - t (float, optional): Time to valuation in years. Default is 0.
    - D (float, optional): Dividend yield of the underlying asset. Default is 0.

    Returns:
    - float: Implied volatility of the option.
    """
    def f(vol):
        return call(T, K, S0, vol, r, t, D) - c

    vol = np.sqrt(2 / T * np.abs(np.log(S0 / (K * np.exp(-r * (T - t))))))  # initial guess
    return fsolve(f, vol)[0]