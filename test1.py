import LSM
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.fft as fft
import numpy as np

stocks_outstanding = 10000
daily_int = 0.0001
daily_div = 0.00015
market = LSM.Market(daily_int, daily_div, stocks_outstanding, "BIST")
market.set_dividend_increment(0.00015)

# for i in range(50, 350, 30):
    # market.add_agent(10, 1000, i, agent_std=0.02)

market.add_agent(25, 1000, 10, agent_std=0.03)
market.add_agent(25, 1000, 26, agent_std=0.03)
market.add_agent(25, 1000, 55, agent_std=0.03)
market.add_agent(25, 1000, 120, agent_std=0.03)

# market.add_agent(30, 1000, 120)

init_hist = norm.rvs(loc=0.0001, scale=0.000125, size=400)
# init_hist[-1] = 2.5
# print("Init Hist:",init_hist, "\n")
init_price = 4

results = market.simulate(0.3, init_hist, init_price, 5000, callback=True, inc_div=True)
market.save_results(results, "four_agent_std_0_03_5000_")
zero_index = int(np.where(results[0] == 0)[0])-1
time = results[0][zero_index:]
prices = results[1][zero_index:]
volume = results[2][zero_index:]

zero_pad_len = 1000*len(time)
fft_prices = fft.fft(prices, zero_pad_len)/(zero_pad_len)
fft_freq = fft.fftfreq(zero_pad_len)
fft_interval = zero_pad_len//2

plt.figure()
plt.subplot(211)
plt.semilogy(time, prices)
plt.ylabel("Price")
plt.xlabel("Time (days)")
plt.subplot(212)
plt.semilogy(time, volume)
plt.ylabel("Volume")
plt.xlabel("Time (days)")

plt.figure()
plt.plot(fft_freq[:fft_interval], abs(fft_prices)[:fft_interval])

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


# prices = np.arange(0.1, 100, 1)
# k = np.array([market.get_collective_demand(i, results[1][-1], results[3], len(results[3])) for i in prices])
# plt.semilogy(prices, abs(k-market.num_stocks))
# # plt.semilogy(prices, shares_os*np.ones(len(prices)))
# plt.show()

# l = market.get_eq_price(results[1][-1], results[3], len(results[3]))
# for a in market.AgentsList:
#     a.demand = a.get_demand(l[0], results[1][-1], 0.00001, 0.0001, results[3], len(results[3]))
# print(l)
# # print(results[3])
# print("Demand:", np.sum([k.demand for k in market.AgentsList]))
# print(market.AgentsList)



###########################################################################################################



# for agent in market.AgentsList:
#     print(agent.money)

# agent1 = LSM.Agent(1000, 10, "Bob")
# prices = np.arange(0.1, 8, 0.1)
# demand1 = [agent1.get_demand(k, 4, 0.003, 0.01, init_hist, (len(init_hist))) for k in prices]
# demand2 = agent1.get_demand(100, 100, 0.003, 0.01, init_hist, len(init_hist))
# print(agent1.stocks, agent1.demand)
# plt.semilogy(prices, demand1)
# plt.show()    



# m2 = LSM.Market(0.01, 0.0001, )