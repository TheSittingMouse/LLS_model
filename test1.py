# Author: Ozan Cem Baş 
# Date: 23.12.2023

import LLS
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.fft as fft
import numpy as np

# Initializing the market
stocks_outstanding = 10000
daily_int = 0.0001
daily_div = 0.00015
market = LSM.Market(daily_int, daily_div, stocks_outstanding, "BIST")
market.set_dividend_increment(0.00015)

# Adding the desired amount of agent to the market
market.add_agent(25, 1000, 10, agent_std=0.03)
market.add_agent(25, 1000, 26, agent_std=0.03)
market.add_agent(25, 1000, 55, agent_std=0.03)
market.add_agent(25, 1000, 120, agent_std=0.03)

# Setting the initial price and the initial history of the market
init_hist = norm.rvs(loc=0.0001, scale=0.000125, size=400)
init_price = 4

# Starting the simulation.
results = market.simulate(0.3, init_hist, init_price, 5000, callback=True, inc_div=True)

# Results can be saved if desired
market.save_results(results, "four_agent_std_0_03_5000_")

# Processing the results to show only the forward time values
zero_index = int(np.where(results[0] == 0)[0])-1
time = results[0][zero_index:]
prices = results[1][zero_index:]
volume = results[2][zero_index:]

# Fourier transformation of the prices
zero_pad_len = 1000*len(time)
fft_prices = fft.fft(prices, zero_pad_len)/(zero_pad_len)
fft_freq = fft.fftfreq(zero_pad_len)
fft_interval = zero_pad_len//2

# Plotting the price and the daily volume
plt.figure()
plt.subplot(211)
plt.semilogy(time, prices)
plt.ylabel("Price")
plt.xlabel("Time (days)")
plt.subplot(212)
plt.semilogy(time, volume)
plt.ylabel("Volume")
plt.xlabel("Time (days)")

# Plotting the FFT results
plt.figure()
plt.plot(fft_freq[:fft_interval], abs(fft_prices)[:fft_interval])
plt.xlabel("Frequency (1/days)")
plt.ylabel("Amplitude")

# Plotting the wealth distributions for inverstors with different memory lengths
legend_names_1 = []
plt.figure()
for memory in market.wealth_dist:
    legend_names_1.append(str(memory))
    plt.plot(market.wealth_dist[memory])
plt.legend(legend_names_1)

legend_names_2 = []
plt.figure()
frac_wealth = market.get_frac_wealth()
for memory in frac_wealth:
    legend_names_2.append(str(memory))
    plt.plot(abs(frac_wealth[memory]))
plt.legend(legend_names_2)
plt.show()
