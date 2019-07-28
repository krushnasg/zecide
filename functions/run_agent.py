from agent import *
from enum import Enum, auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from dataset.read_Data import df
# import importlib
# agent = importlib.import_module('agent')

#Set data file
# df = pd.read_csv('dataset/GOOG-year.csv')

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

class Agents(AutoName):
    turtle_agent = auto()
    moving_average_agent = auto()
    signal_rolling_agent = auto()
    policy_gradient_agent = auto()
    q_learning_agent = auto()
    evolution_strategy_agent = auto()
    double_q_learning_agent = auto()
    recurrent_q_learning_agent = auto()
    double_recurrent_q_learning_agent = auto()
    duel_q_learning_agent = auto()
    double_duel_q_learning_agent = auto()
    duel_recurrent_q_learning_agent = auto()
    double_duel_recurrent_q_learning_agent = auto()
    actor_critic_agent = auto()
    actor_critic_duel_agent = auto()
    actor_critic_recurrent_agent = auto()
    actor_critic_duel_recurrent_agent = auto()
    curiosity_q_learning_agent = auto()
    recurrent_curiosity_q_learning_agent = auto()
    duel_curiosity_q_learning_agent = auto()
    neuro_evolution_agent = auto()
    neuro_evolution_novelty_search_agent = auto()
    abcd_strategy_agent = auto()
    
#choose agent
chosenAgent = Agents.actor_critic_recurrent_agent

def run_agent(agent_name,data):
    eval('run_' + agent_name.name)(data)

def run_turtle_agent(df):
    count = int(np.ceil(len(df) * 0.1))
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['trend'] = df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1

    states_buy, states_sell, total_gains, invest = turtle_agent.buy_stock(df.Close, signals['signal'])

    close = df['Close']
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest



    
def run_moving_average_agent(df):
    short_window = int(0.025 * len(df))
    long_window = int(0.05 * len(df))

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    signals['short_ma'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_ma'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] 
                                                > signals['long_ma'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()

    states_buy, states_sell, total_gains, invest = moving_average_agent.buy_stock(df.Close, signals['positions'])

    close = df['Close']
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()
    return states_buy, states_sell, total_gains, invest


def run_signal_rolling_agent(df):
    states_buy, states_sell, total_gains, invest = signal_rolling_agent.buy_stock(df.Close, initial_state = 1, 
                                                         delay = 4, initial_money = 10000)
    close = df['Close']
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest






def run_policy_gradient_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    agt = policy_gradient_agent.Agent(state_size = window_size,
                window_size = window_size,
                trend = close,
                skip = skip)
    agt.train(iterations = 200, checkpoint = 10, initial_money = initial_money)


    states_buy, states_sell, total_gains, invest = agt.buy(initial_money = initial_money)

    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest





def run_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip, 
                batch_size = batch_size)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)


    # In[5]:


    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    # In[6]:


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_evolution_strategy_agent(df):
    close = df.Close.values.tolist()
    window_size = 30
    skip = 1
    initial_money = 10000

    model = evolution_strategy_agent.Model(input_size = window_size, layer_size = 500, output_size = 3)
    agent = evolution_strategy_agent.Agent(model = model, 
                window_size = window_size,
                trend = close,
                skip = skip,
                initial_money = initial_money)
    agent.fit(iterations = 500, checkpoint = 10)


    # In[7]:


    states_buy, states_sell, total_gains, invest = agent.buy()


    # In[8]:


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest 


def run_double_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = double_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)



    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)

    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest

def run_recurrent_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = recurrent_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)

    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_double_recurrent_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = double_recurrent_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)

    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_duel_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = duel_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip, 
                batch_size = batch_size)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest 


def run_double_duel_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_duel_recurrent_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = duel_recurrent_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest

def run_double_duel_recurrent_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = double_duel_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()
    
    return states_buy, states_sell, total_gains, invest



def run_actor_critic_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = actor_critic_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_actor_critic_duel_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = actor_critic_duel_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)


    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_actor_critic_recurrent_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = actor_critic_recurrent_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest

def run_actor_critic_duel_recurrent_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = actor_critic_duel_recurrent_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest



def run_curiosity_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = curiosity_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest

def run_recurrent_curiosity_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = recurrent_curiosity_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_duel_curiosity_q_learning_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1
    batch_size = 32
    agent = duel_curiosity_q_learning_agent.Agent(state_size = window_size, 
                window_size = window_size, 
                trend = close, 
                skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return     states_buy, states_sell, total_gains, invest

def run_neuro_evolution_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1

    population_size = 100
    generations = 100
    mutation_rate = 0.1
    neural_evolve = neuro_evolution_agent.NeuroEvolution(population_size, mutation_rate, neuralnetwork,
                                window_size, window_size, close, skip, initial_money)

    fittest_nets = neural_evolve.evolve(50)

    states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest

def run_neuro_evolution_novelty_search_agent(df):
    close = df.Close.values.tolist()
    initial_money = 10000
    window_size = 30
    skip = 1

    novelty_search_threshold = 6
    novelty_log_maxlen = 1000
    backlog_maxsize = 500
    novelty_log_add_amount = 3

    population_size = 100
    generations = 100
    mutation_rate = 0.1
    neural_evolve = neuro_evolution_novelty_search_agent.NeuroEvolution(population_size, mutation_rate, neuralnetwork,
                                window_size, window_size, close, skip, initial_money)

    fittest_nets = neural_evolve.evolve(100)

    states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)


    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest


def run_abcd_strategy_agent(df):
    signal = abcd_strategy_agent.abcd(df['Close'])
    states_buy, states_sell, total_gains, invest, states_money = abcd_strategy_agent.buy_stock(df.Close, signal)

    close = df['Close']
    fig = plt.figure(figsize = (15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()


    fig = plt.figure(figsize = (15,5))
    plt.plot(states_money, color='r', lw=2.)
    plt.plot(states_money, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(states_money, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.legend()
    plt.show()

    return states_buy, states_sell, total_gains, invest, states_money
    


run_agent(chosenAgent,df)
