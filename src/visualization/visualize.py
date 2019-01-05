# plotting setup
import logging
import pandas as pd
import plotly.plotly as py
import plotly.offline as pyoffline
import plotly.graph_objs as go
import cufflinks
import plotly.tools as pytools

cufflinks.set_config_file(offline=True, world_readable=False, theme='ggplot')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

def plot_rewards(df, filename=None):
    """
    Plot average reward per time step for greedy, epsilon greedy and decaying epsilon greedy policies
    The input dataframe should have one 'reward' column for each of the policies
    param df: Dataframe with rewards and actions for all policies
    param filename: If None, uses iplot to plot inline in jupyter nb. If not none, plot is written to file
    """
    reward_columns = [col for col in df.columns if 'reward' in col]

    rewards_plot_data = [go.Scatter(x=df.index, y=df[col], name=col) for col in reward_columns]

    layout = go.Layout(title="Cumulative Average reward recieved over time")
    fig = go.Figure(rewards_plot_data, layout=layout)
    logging.info("Plotting rewards")
    if filename is None:
        pyoffline.iplot(fig)
    else:
        pyoffline.plot(fig, filename=filename)


def plot_actions(df, filename=None):
    """
    Plot average reward per time step for greedy, epsilon greedy and decaying epsilon greedy policies
    The input dataframe should have one 'action' column for each of the policies
    param df: Dataframe with rewards and actions for all policies
    param filename: If None, uses iplot to plot inline in jupyter nb. If not none, plot is written to file
    """
    fig = pytools.make_subplots(rows=1, cols=3, shared_xaxes=True)
    fig['layout'].update(title="Actions taken by different agents")
    action_columns = [col for col in df.columns if 'action' in col]
    for idx, col in enumerate(action_columns):
        trace = go.Scatter(x=df.index, y=df[col], name=col, mode='markers')
        fig.append_trace(trace, 1, idx+1)
    pyoffline.plot(fig, filename=filename)
