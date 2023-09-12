import pandas as pd
import numpy as np


class StanfordDroneAgent:

    def __init__(self, agent_df: pd.DataFrame):
        self.id = agent_df["Id"].unique().item()
        self.label = agent_df.iloc[0].loc["label"]
        self.timesteps = agent_df["frame"].to_numpy()

        self.fulltraj = np.empty((len(self.timesteps), 2))
        self.fulltraj.fill(np.nan)
        self.fulltraj[:, 0] = agent_df["x"].values.flatten()
        self.fulltraj[:, 1] = agent_df["y"].values.flatten()

        assert not np.isnan(self.fulltraj).any()

    def get_traj_section(self, time_window: np.array) -> np.array:
        return self.fulltraj[np.in1d(self.timesteps, time_window), :]

    def position_at_timestep(self, timestep: int) -> np.array:
        return self.fulltraj[np.where(self.timesteps == timestep)].reshape(2,)

    def get_data_availability_mask(self, time_window: np.array) -> np.array:
        return np.in1d(time_window, self.timesteps).astype(float)

    def get_travelled_distance(self, time_window: np.array) -> float:
        traj_section = self.get_traj_section(time_window=time_window)
        diffs = traj_section[1:, ...] - traj_section[:-1, ...]
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.sum(dists))
