"""Actor-critic network. The actor is deliberately tiny (OBS -> 64 -> 64 -> 3 + 5) so
that it can run inside LPSim's per-vehicle CUDA kernel with weights in constant memory.
The critic is used only during training."""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import NUM_LAT, NUM_LON, OBS_DIM


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.actor = _mlp(OBS_DIM, hidden, NUM_LAT + NUM_LON)
        self.critic = _mlp(OBS_DIM, hidden, 1)
        with torch.no_grad():   # near-uniform initial policy, small value outputs
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()
            self.critic[-1].weight.mul_(0.1)

    def dists(self, obs, lat_mask):
        logits = self.actor(obs)
        lat = logits[..., :NUM_LAT].masked_fill(~lat_mask, -1e9)
        # validate_args=False: argument validation would force a GPU->CPU sync on every call
        return (Categorical(logits=lat, validate_args=False),
                Categorical(logits=logits[..., NUM_LAT:], validate_args=False))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, lat_mask, deterministic=False):
        d_lat, d_lon = self.dists(obs, lat_mask)
        if deterministic:
            lat, lon = d_lat.probs.argmax(-1), d_lon.probs.argmax(-1)
        else:
            lat, lon = d_lat.sample(), d_lon.sample()
        return lat, lon, d_lat.log_prob(lat) + d_lon.log_prob(lon)
