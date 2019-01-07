"""
Short description - This module simulates a 10 armed bandit problem using a simple tabular actioin-value method

:copyright: 2018 Manifold Inc.
:author: Rajendra Koppula <rkoppula@manifold.ai>
"""
import gym
import gym_bandits
import logging
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from src.visualization.visualize import plot_rewards, plot_actions

N_STEPS_PER_PLAY = 3000

class Agent():
    def __init__(self, env, start_epsilon, end_epsilon=None):
        self.env = env
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon

    @staticmethod
    def get_greedy_action(df):
        if len(df) != 0:
            Q = df.groupby("a").mean()["r"]
            # Some actions may not have been taken so far. Reindex to add them to the mix
            Q = Q.reindex(range(10), fill_value=0.0)
            # If there are multiple actions with values equal to maximum, random sample one
            maxs = Q[Q==Q.max()]
            if len(maxs) > 1:
                return maxs.sample().index[0]
            else:
                return Q.idxmax()
        else:
            # If empty, choose a random action
            return np.random.randint(10)

    def play(self, n_steps):
        # Action, reward and epsilon for each step
        df = pd.DataFrame(np.zeros((n_steps, 3)), columns=["a", "r", "e"], index=range(n_steps))
        df['a'] = df.a.astype(int)

        # Setup up epsilon decay
        if self.end_epsilon is None:
            # No decay
            df["e"] = np.linspace(self.start_epsilon, self.start_epsilon, n_steps)
        else:
            df["e"] = np.linspace(self.start_epsilon, self.end_epsilon, n_steps)

        # Run for n_steps
        # At each step, make a decision to take a random action or a greedy action
        # Greedy => select the action with the highest estimate of the Expected return
        for step in range(n_steps):
            if df.loc[step, "e"] == 0:
                # greedy policy
                # use history upto step-1
                action = Agent.get_greedy_action(df.loc[:step-1])
            else:
                # epsilon-greedy policy
                e_sample = np.random.random()
                if e_sample <= df.loc[step, "e"]:
                    # Explore by taking a random action
                    action = np.random.randint(0, 10)
                else:
                    # Exploit current knowledge about the system
                    action = Agent.get_greedy_action(df.loc[:step-1])

            _, r, _, _ = self.env.step(action)
            df.loc[step, "r"] = r
            df.loc[step, "a"] = action

        return df


def play_wrapper(name, env, agent):
    avg_r_col = name+' avg reward'
    action_col = name+" action"
    columns = [avg_r_col, action_col]

    df = agent.play(n_steps=N_STEPS_PER_PLAY)
    # Compute average reward recieved up until every step
    df[avg_r_col] = df.r.cumsum()/(df.index + 1)
    df = df.rename(columns={"a":action_col})
    # df = df[columns]

    return df

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    env = gym.make("BanditTenArmedGaussian-v0")
    env.reset()

    # epsilon = 0 => No exploration, always greedy
    agent = Agent(env, 0)
    logging.info("Training greedy agent")
    g_df = play_wrapper('greedy agent', env, agent)

    # Note: Since the steps are IID in 10-armed bandit, we continue to use the same env without resetting it This allows
    # us to evaluate different agents consistently on the same problem.  Resetting env would change the true mean
    # returns for each action. In this case, to properly evaluate we would need to average across many plays for each
    # agent to compare them.
    logging.info("Training epsilon-greedy agent")
    agent = Agent(env, 0.1)
    epsilon_g_df = play_wrapper('epsilon-greedy agent', env, agent)

    # decaying epsilon
    logging.info("Training decaying epsilon-greedy agent")
    agent = Agent(env, 0.1, 0.01)
    decaying_epsilon_g_df = play_wrapper('decaying epsilon-greedy agent', env, agent)

    df = pd.concat([g_df, epsilon_g_df, decaying_epsilon_g_df], axis=1)

    rewards_plot_filename = os.path.join(os.environ['PROJECT_DIR'], "reports/figures/10_armed_agents_rewards.html")
    plot_rewards(df, filename=rewards_plot_filename)

    actions_plot_filename = os.path.join(os.environ['PROJECT_DIR'], "reports/figures/10_armed_agents_actions.html")
    plot_actions(df, filename=actions_plot_filename)
