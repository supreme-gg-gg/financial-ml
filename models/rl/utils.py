import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd, numpy as np

def get_data(file_path, scale=True):

    '''
    Obtains and preprocesses the data from the file path.
    Performs regularisation if scale is set to True.
    Currently uses only the 'Adj Close' and 'Volume' columns.
    Computes the return of the stock.
    '''

    # TODO: Add technical indicators in processing data

    df = pd.read_csv(file_path)
    df = df[['Adj Close', 'Volume']]
    df.Volume.replace(0,1,inplace=True)
    df['Return'] = (df['Adj Close'] - df['Adj Close'].shift()) / df['Adj Close'].shift()
    R = df.Return
    if scale:
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        df = (df - np.array(mean)) / np.array(std)
    df['Return'] = R # Return is not scaled
    # min_values = df.min(axis=0)
    # max_values = df.max(axis=0)
    data = df
    return data

def plot_durations(episode_durations, show_result=False):
    
    '''
    Plot the duration of each episode, 
    as well as the mean duration of the last 100 episodes.

    Args:
    - episode_durations (list): list of durations of each episode
    - show_result (bool): whether to show the result or not

    Returns:
    - None
    '''
    
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())