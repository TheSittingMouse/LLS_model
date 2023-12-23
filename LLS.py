import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt

# need to work on the dividents !!!!
# dividents need to be time dependent
# for now, use constant divident
class Market:
    def __init__(self, daily_interest, daily_divident, num_stocks, name=None):
        self.daily_interest = daily_interest
        self.daily_divident = daily_divident
        self.dividend_increment = 0
        self.num_stocks = num_stocks
        self.wealth_dist = {}
        self.AgentsList = []
        self.agent_number = 0
        self.name = name

    def __repr__(self):
        print(f"LLS Model Object {self.name}:")
        print("\t Daily interest:", self.daily_interest)
        print("\t Daily divident:", self.daily_divident)
        print("\t Agents:", [i.name for i in self.AgentsList])
        return ""
    
    def set_dividend_increment(self, increment):
        self.dividend_increment = increment
        return 0

    def add_agent(self, number, wealth, memory, agent_std=0.01, name="Undefined"):
        if number == -1:
            agent = Agent(wealth, memory, name)
            agent.agent_std = agent_std
            self.AgentsList.append(agent)

            key_list = [key for key in self.wealth_dist]
            if memory in key_list:
                self.wealth_dist[memory][0] += wealth
            elif memory not in key_list:
                self.wealth_dist[memory] = [wealth]

            return 0
        for i in range(number):
            agent = Agent(wealth, memory, self.agent_number)
            agent.agent_std = agent_std
            self.AgentsList.append(agent)

            key_list = [key for key in self.wealth_dist]
            if memory in key_list:
                self.wealth_dist[memory][0] += wealth
            elif memory not in key_list:
                self.wealth_dist[memory] = [wealth]

            self.agent_number += 1
        return 0

    def get_collective_demand(self, Ph, Pt, hist, time):
        N = 0
        for agent in self.AgentsList:
            N += agent.get_demand(Ph, Pt, self.daily_divident, self.daily_interest, hist, time)
        return N
    
    def get_eq_price(self, Pt, hist, time, lower_lim=1e-5,  upper_lim=1000):
        def f(x):
            return (self.get_collective_demand(x, Pt, hist, time) - self.num_stocks)
        
        a = lower_lim
        b = Pt + upper_lim

        while True:
            try:
                Ph = opt.brenth(f, a, b)
                break
            except:
                a = b
                b += 1000

        demand = self.get_collective_demand(Ph, Pt, hist, time)
        return [Ph, demand]        
    
    def update_collective_wealth(self):
        for key in self.wealth_dist:
            self.wealth_dist[key].append(0)
        for agent in self.AgentsList:
            self.wealth_dist[agent.memory][-1] += agent.wealth
        return 0
    
    def get_frac_wealth(self):
        total_wealth = 0
        frac_list = {}
        for key in self.wealth_dist:
            total_wealth += np.array(self.wealth_dist[key])
        for key in self.wealth_dist:
            frac_list[key] = self.wealth_dist[key]/total_wealth
        return frac_list

    def simulate(self, init_dist, init_hist, init_price, time, initialize=True, callback=False, inc_div=False):
        len_hist = len(init_hist)
        hist = np.hstack((np.array(init_hist),np.zeros(((time-1),))))
        time_array = np.arange((-len_hist+1),(time))
        price = np.zeros(((time+len_hist-1),))
        volume = np.zeros(((time+len_hist-1),))
        price[len_hist-1] = init_price

        # initialization
        if initialize:
            for agent in self.AgentsList:
                agent.stocks = (agent.wealth*init_dist)/init_price
            tot_stocks = np.sum([i.stocks for i in self.AgentsList])
            if tot_stocks >= self.num_stocks:
                print("WARNING: The total number of stocks held by investors initially cannot be greater than shares outstanding. Please change your initial conditions. \n")
                for agent in self.AgentsList:
                    agent.stocks = 0
                return -1

        # simulating for all time values
        for t in range((len_hist-1),(time+len_hist-2)):
            # get the new price of the asset at time t+1
            price_demand = self.get_eq_price(price[t], hist, t)

            demand = price_demand[1]
            price[t+1] = price_demand[0]
            hist[t+1] = (price[t+1]-price[t]+self.daily_divident)/price[t]

            # update the assets of the investers
            tot_wealth = 0
            for agent in self.AgentsList:
                agent.demand = agent.get_demand(price[t+1], price[t], self.daily_divident, self.daily_interest, hist, t)
                volume[t+1] += np.abs(agent.stocks-agent.demand)
                agent.wealth += (agent.stocks*self.daily_divident)+((agent.wealth-agent.stocks*price[t])*self.daily_interest) + agent.stocks*(price[t+1]-price[t])
                agent.stocks = agent.demand
                tot_wealth += agent.wealth
                agent.get_new_random()
            
            self.update_collective_wealth()

            if inc_div:
                self.daily_divident += self.dividend_increment

            if callback:
                print("Simulation at time:",(t-len_hist+2),"|","Price:",price[t+1],"|", "Volume:", volume[t+1], "|", "Demand:", demand,"|", "Wealth:", tot_wealth)

        return [time_array,price,volume,hist]
    
    def save_results(self, result, name):
        import pandas as pd
        import os

        # Define folder name
        folder_name = name

        # Create the folder
        no_folder = True
        n = 1
        while no_folder:
            try:
                os.makedirs(folder_name)
                break
            except OSError as e:
                # if e.errno != e.errno.EEXIST:
                #     raise
                print("Folder name already exists. Assigning new name.")
                folder_name = str(name + str(n))
                n += 1
        
        agent_lib = self.get_frac_wealth()
        for key in agent_lib:
            agent_lib[key] = []
        for agent in self.AgentsList:
            mem = agent.memory
            agent_lib[mem].append(agent)

        wealth_lib = self.get_frac_wealth()
        result_lib = {}
        info_lib = {"Name":[self.name], "Interest":[self.daily_interest], "Dividend":[self.daily_divident], "StocksOS":[self.num_stocks]}
        result_lib["Time"] = result[0]
        result_lib["Price"] = result[1]
        result_lib["Volume"] = result[2]
        result_lib["History"] = result[3]

        agent_db = pd.DataFrame(agent_lib)#.to_csv(str("agents.csv"))
        wealth_db = pd.DataFrame(wealth_lib)#.to_csv(str("wealth.csv"))
        info_db = pd.DataFrame(info_lib)#.to_csv(str("info.csv"))
        results_db = pd.DataFrame(result_lib)#.to_csv("results.csv")

        dataframes = [agent_db, wealth_db, info_db, results_db]
        filenames = ["agents.csv", "wealth.csv", "info.csv", "results.csv"]

        for df, filename in zip(dataframes, filenames):
            filepath = os.path.join(folder_name, filename)
            print(filepath)
            df.to_csv(filepath, index=False)  # Save dataframe as CSV file

        return 0

class Agent:
    def __init__(self, wealth, memory, name=None):
        self.wealth = wealth
        self.stocks = 0
        self.memory = memory
        self.name = name
        self.demand = 0
        self.agent_std = 0.01
        self.agent_mean = 0
        self.agent_param = random.gauss(self.agent_mean, self.agent_std)
        self.wealth_array = []

    def __repr__(self):
        print(f"Agent Object {self.name} of Market:")
        print("\t wealth:", self.wealth)
        print("\t Memory length:", self.memory) 
        print("\t Number of stocks:", self.stocks)
        print("\t Demand:", self.demand)
        return ""
    
    def get_new_random(self):
        self.agent_param = random.gauss(self.agent_mean, self.agent_std)

    def get_util(self, Xi, Ph, Pt, D, r, hist, time):
        """
        Returns the 
        """
        mem = self.memory
        hist_list = hist[time-mem: time+1]
        Wh = self.wealth + self.stocks*D + (self.wealth-self.stocks*Pt)*r + self.stocks*(Ph-Pt)
        if Wh < 1e-5:
            Wh = 1e-5
        # print("Memory Hist:",hist_list)
        # print(np.mean(np.log(abs((1-Xi)*Wh*(1+r)+Xi*Wh*(1+hist_list)))))

        util = np.mean(np.log((1-Xi)*Wh*(1+r)+Xi*Wh*(1+hist_list)))

        return [util, Wh]

    def get_demand(self, Ph, Pt, D, r, hist, time):
        """
        Gets and updates the demand of the agent for the specified parameters at a time t

        Parameters:
        Ph (float): The future price
        Pt (float): The current price
        D (float): The daily divident rate
        r (float): Teh daily interest rate for bonds
        hist (array): An array of past returns of the asset
        time (int): The time at which the demand is calculated

        Output (float): the demand for the given future price
        """

        # maximizer

        n = 1
        n_max = 10
        Xi_list = [0., 0.5, 1.]
        arg_Xi = 0

        while n<n_max:
            util_Wh0 = self.get_util(Xi_list[0], Ph, Pt, D, r, hist, time)
            util_Wh1 = self.get_util(Xi_list[1], Ph, Pt, D, r, hist, time)
            util_Wh2 = self.get_util(Xi_list[2], Ph, Pt, D, r, hist, time)
            util_list = [util_Wh0[0], util_Wh1[0], util_Wh2[0]]
            arg_util_min = np.argmin(util_list)
            arg_Xi = np.argmax(util_list)
            Xi_list[arg_util_min] = (np.sum(Xi_list)-Xi_list[arg_util_min])/2
            n += 1
        Xh = Xi_list[arg_Xi]
        Wh = self.get_util(Xh, Ph, Pt, D, r, hist, time)[1]

        # adding randomness to accont for various interactions
        Xh = Xh + self.agent_param
        if Xh>0.99:
            Xh = 0.99
            Wh = self.get_util(Xh, Ph, Pt, D, r, hist, time)[1]
        elif Xh<0.01:
            Xh = 0.01
            Wh = self.get_util(Xh, Ph, Pt, D, r, hist, time)[1]

        demand = Xh*Wh/Ph
        # if demand*Ph > self.wealth:
        #     return self.wealth/Ph
        # if abs(demand-self.stocks) < 0.01:
        #     demand = 0
        

        # updating the demand attribute of the agent
        # self.demand = Xh*Wh/Ph
        # print(Xh, Ph, demand)
    
        return demand

