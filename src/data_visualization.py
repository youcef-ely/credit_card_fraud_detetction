import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt  # noqa: E401
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pretty_errors
import itertools
from tqdm import tqdm



def check_pandas(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
    return data


def crosstabs(data, columns: list, shape: tuple = (2, 2), figsize: tuple = (10, 10), params: dict = {}):
  combs = list(itertools.combinations(columns, 2))
  assert len(combs) <= shape[0] * shape[1] 
  fig, axs = plt.subplots(*shape, figsize = figsize)
  axs = axs.flatten() if shape != (1, 1) else [axs] 
  for i, comb in enumerate(tqdm(combs)):
    sns.heatmap(pd.crosstab(data[comb[0]], data[comb[1]]), annot = True, fmt = '.0f', **params)
  plt.show()



def barplots(data, columns: list, shape: tuple = (2, 2)):
    data = check_pandas(data)
    assert len(columns) <= shape[0] * shape[1], "Number of columns exceeds the grid shape"

    fig = make_subplots(rows = shape[0], cols = shape[1], subplot_titles = columns)

    for i, col in enumerate(tqdm(columns, desc = 'Plotting barplots'), start = 1):
        temp = data[col].value_counts()
        fig.add_trace(go.Bar(x = temp.index, y = temp.values, name = col), 
                      row = (i - 1) // shape[1] + 1, 
                      col = (i - 1) % shape[1] + 1)
    fig.show()
    return fig




def target_histograms(data, x: list, y: str, shape: tuple):
    assert len(x) <= shape[0] * shape[1], "Number of columns exceeds the grid shape"

    fig = make_subplots(rows = shape[0], cols = shape[1], subplot_titles = x)

    for i, col in enumerate(tqdm(x, desc = f'Plotting histograms grouped by {y} column'), start = 1):
        for cat in data[y].unique():
            fig.add_trace(go.Histogram(x = data[data[y] == cat][col], name = str(cat), histnorm = 'probability density'), 
                          row = (i - 1) // shape[1] + 1, 
                          col = (i - 1) % shape[1] + 1)

    fig.show()
    return fig



def histograms(data, x: list, shape: tuple, params: dict = {}):
    assert len(x) <= shape[0] * shape[1], "Number of columns exceeds the grid shape"
    fig = make_subplots(rows = shape[0], cols = shape[1], subplot_titles = x)
    for i, col_name in enumerate(tqdm(x, desc = 'Plotting histograms'), start = 1):
        row = (i - 1) // shape[1] + 1
        col = (i - 1) % shape[1] + 1
        fig.add_trace(go.Histogram(x = data[col_name], histnorm = 'probability density', name = str(col_name), **params),
                      row = row, col = col)
      

    fig.update_traces(opacity = 0.6)
    fig.show(rerender = 'colab')  
    return fig



def target_boxplots(data, x: list, y: str, shape: tuple):
    assert len(x) <= shape[0] * shape[1], "Number of columns exceeds the grid shape"

    fig = make_subplots(rows = shape[0], cols = shape[1], subplot_titles = x)

    for i, col in enumerate(tqdm(x, desc = 'Plotting boxplots'), start = 1):
        row = (i - 1) // shape[1] + 1
        col_idx = (i - 1) % shape[1] + 1

        fig.add_trace(
            go.Box(y = data[col], x = data[y], name = col, boxmean = True),  # x and y switched
            row = row, col = col_idx
        )

        fig.update_xaxes(title_text = col, row = row, col = col_idx)

    fig.show()


def scatter_plots(data, x: list, y: str, shape: tuple):
    combinations = list(itertools.combinations(x, 2))
    assert len(combinations) <= shape[0] * shape[1]
    
    fig = make_subplots(rows = shape[0], cols = shape[1], subplot_titles = [f"{comb[0]} vs {comb[1]}" for comb in combinations])
    
    for i, comb in enumerate(tqdm(combinations)):
        fig.add_trace(px.scatter(data, x = comb[0], y = y, color = 'is_fraud').data[0], 
                        row = i // shape[1] + 1, 
                        col = i % shape[1] + 1)
    fig.show()
    return fig