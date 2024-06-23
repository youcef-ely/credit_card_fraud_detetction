import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt  # noqa: E401
import pretty_errors
import itertools
from tqdm import tqdm 





def crosstabs(data, columns: list, shape: tuple = (2, 2), figsize: tuple = (10, 10), params: dict = {}):
    combs = list(itertools.combinations(columns, 2))
    assert len(combs) <= shape[0] * shape[1] 
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs] 
    for i, comb in enumerate(tqdm(combs), desc = 'Plotting crosstabs'):
        sns.heatmap(pd.crosstab(data[comb[0]], data[comb[1]]), annot = True, fmt = '.0f', **params)
    plt.show()




def barplots(data, columns: list, shape: tuple = (2, 2), figsize: tuple = (10, 20), params: dict = {}):
    assert len(columns) <= shape[0] * shape[1]
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs]
    for i, col in enumerate(tqdm(columns, desc = 'Plotting barplots')):
        temp = data[col].value_counts()
        sns.barplot(x = temp.index, y = temp.values, ax = axs[i], **params)
        plt.title(col)
    plt.show()




def histograms(data, shape: tuple, y: str = None, figsize: tuple = (20, 10), params: dict = {}):
    x = [col for col in data.columns if col != y]
    assert len(x) <= shape[0] * shape[1]
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs]
    
    if y:
        for i, col in enumerate(tqdm(x, desc = f'Plotting histograms grouped by {y}')):
            for cat in data[y].unique():
                sns.histplot(data[data[y] == cat][col], label = str(cat), ax = axs[i], **params)
            axs[i].legend()
    else:
        for i, col in enumerate(tqdm(x, desc = 'Plotting histograms of columns')):
            sns.histplot(data = data, x = col, ax = axs[i], **params)
            axs[i].legend()
    
    plt.show()



def kdeplots(data, shape: tuple, y: str = None, figsize: tuple = (20, 10), params: dict = {}):
    x = [col for col in data.columns if col != y]
    assert len(x) <= shape[0] * shape[1]
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs]
    
    if y:
        for i, col in enumerate(tqdm(x, desc = f'Plotting histograms grouped by {y}')):
            for cat in data[y].unique():
                sns.kdeplot(data[data[y] == cat][col], label = str(cat), ax = axs[i], **params)
            axs[i].legend()
    else:
        for i, col in enumerate(tqdm(x, desc = 'Plotting histograms of columns')):
            sns.kdeplot(data = data, x = col, ax = axs[i], **params)
            axs[i].legend()
    
    plt.show()



def target_boxplots(data, x: list, y: str, shape: tuple, figsize: tuple, params: dict = {}):
    assert len(x) <= shape[0] * shape[1]
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs]
    unique_vals = data[y].unique()
    for i, col in enumerate(x):
        data_to_plot = [data[data[y] == cat][col] for cat in unique_vals]
        axs[i].boxplot(data_to_plot, labels = unique_vals)
        axs[i].set_title(col)
        plt.legend()  
    plt.show()


def scatter_plots(data, x: list, y: str, shape: tuple, figsize: tuple):
    combinations = list(itertools.combinations(x, 2))
    assert len(combinations) <= shape[0] * shape[1]
    fig, axs = plt.subplots(*shape, figsize = figsize)
    axs = axs.flatten() if shape != (1, 1) else [axs]
    for i, comb in enumerate(combinations):
        groups = data.groupby(y)[list(comb)]
        for name, group in groups:
            axs[i].plot(group[comb[0]], group[comb[1]], label = name, marker = 'o', linestyle = '', ms = 2)
            axs[i].set_xlabel(comb[0])
            axs[i].set_ylabel(comb[1])
            axs[i].legend()
    plt.show()    