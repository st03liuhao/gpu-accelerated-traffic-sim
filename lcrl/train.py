"""PPO training with parameter sharing: every vehicle in every segment is an agent.

Samples come from freeway segments (lcrl/env.py) and complete intersections
(lcrl/ix_env.py) at the same time; both expose the same observation layout.

    python -m lcrl.train --iterations 400 --out runs/lc_ppo
"""
import argparse
import dataclasses
import json
import os
import time

import torch

from .config import NUM_LAT, OBS_DIM, TrainConfig
from .env import S, LaneChangeEnv
from .graph import GraphedEnv
from .ix_env import IntersectionEnv
from .policy import ActorCritic


class MultiEnv:
    """Several batched envs presented as one flat batch of agent slots."""

    def __init__(self, envs):
        self.envs = envs
        self.sizes = [e.E * e.N for e in envs]
        self.M = sum(self.sizes)

    def observe(self):
        outs = [e.observe() for e in self.envs]
        return (torch.cat([o.reshape(-1, OBS_DIM) for o, _, _ in outs]),
                torch.cat([m.reshape(-1, NUM_LAT) for _, m, _ in outs]),
                torch.cat([v.reshape(-1) for _, _, v in outs]))

    def step(self, lat, lon):
        rs, ds = [], []
        for e, la, lo in zip(self.envs, lat.split(self.sizes), lon.split(self.sizes)):
            r, d = e.step(la.view(e.E, e.N), lo.view(e.E, e.N))
            rs.append(r.reshape(-1))
            ds.append(d.reshape(-1))
        return torch.cat(rs), torch.cat(ds)


def train(tc: TrainConfig, out_dir, device="cuda", log_every=5, init=None):
    torch.manual_seed(tc.seed)
    os.makedirs(out_dir, exist_ok=True)
    envs = {"freeway": LaneChangeEnv(tc.env, tc.reward, device=device, seed=tc.seed)}
    if tc.use_ix:
        envs["intersection"] = IntersectionEnv(tc.ix, tc.reward, device=device, seed=tc.seed + 1)
    env = MultiEnv([GraphedEnv(e) for e in envs.values()])   # each decision replays one CUDA graph per env
    net = ActorCritic(tc.hidden).to(device)
    if init:                                  # fine-tune from an earlier policy (same obs / actions)
        net.load_state_dict(torch.load(init, map_location=device)["state_dict"])
    opt = torch.optim.Adam(net.parameters(), lr=tc.lr, eps=1e-5)
    Tn, M = tc.rollout_steps, env.M
    z = lambda *shape, **kw: torch.zeros(Tn, M, *shape, device=device, **kw)
    buf = dict(obs=z(OBS_DIM), mask=z(NUM_LAT, dtype=torch.bool), lat=z(dtype=torch.long),
               lon=z(dtype=torch.long), logp=z(), val=z(), rew=z(), done=z(), valid=z(dtype=torch.bool))
    history = []
    obs, mask, valid = env.observe()
    t_start = time.time()
    for it in range(1, tc.iterations + 1):
        frac = 1.0 - (it - 1) / tc.iterations
        for g in opt.param_groups:
            g["lr"] = tc.lr * frac
        if it % 10 == 1:   # stats accumulate over 10-iteration windows (trips outlast a rollout)
            for e in env.envs:
                e.stats.zero_()
        # ---------------- rollout
        with torch.no_grad():
            for t in range(Tn):
                lat, lon, logp = net.act(obs, mask)
                buf["obs"][t], buf["mask"][t], buf["valid"][t] = obs, mask, valid
                buf["lat"][t], buf["lon"][t], buf["logp"][t] = lat, lon, logp
                buf["val"][t] = net.value(obs)
                r, done = env.step(lat, lon)
                buf["rew"][t], buf["done"][t] = r, done.float()
                obs, mask, valid = env.observe()
            # a slot that was empty contributes nothing; treat it as terminal
            last_val = net.value(obs) * valid
            adv = torch.zeros_like(buf["rew"])
            gae = torch.zeros(M, device=device)
            for t in reversed(range(Tn)):
                nv = last_val if t == Tn - 1 else buf["val"][t + 1]
                nonterm = 1.0 - buf["done"][t]
                v_ok = buf["valid"][t].float()
                delta = buf["rew"][t] + tc.gamma * nv * nonterm - buf["val"][t]
                gae = (delta + tc.gamma * tc.gae_lambda * nonterm * gae) * v_ok
                adv[t] = gae
            ret = adv + buf["val"]
        # ---------------- PPO update on valid samples only
        sel = buf["valid"]
        b = {k: v[sel] for k, v in buf.items() if k != "valid"}
        b_adv, b_ret = adv[sel], ret[sel]
        n = b_adv.shape[0]
        clipfracs, losses = [], []
        for _ in range(tc.epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, tc.minibatch):
                idx = perm[i:i + tc.minibatch]
                d_lat, d_lon = net.dists(b["obs"][idx], b["mask"][idx])
                logp = d_lat.log_prob(b["lat"][idx]) + d_lon.log_prob(b["lon"][idx])
                ratio = (logp - b["logp"][idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - tc.clip, 1 + tc.clip)).mean()
                v = net.value(b["obs"][idx])
                vl = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                ent = (d_lat.entropy() + d_lon.entropy()).mean()
                loss = pg + tc.vf_coef * vl - tc.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), tc.max_grad_norm)
                opt.step()
                # kept on the GPU; read back once per iteration
                clipfracs.append(((ratio - 1).abs() > tc.clip).float().mean().detach())
                losses.append(torch.stack([pg.detach(), vl.detach(), ent.detach()]))
        # ---------------- logging
        # one device-to-host transfer per iteration
        ls = torch.stack(losses).mean(0)
        host = torch.cat([b["rew"].mean()[None], ls, torch.stack(clipfracs).mean()[None]]
                         + [e.stats.sum(0) for e in envs.values()]).tolist()
        row = dict(it=it, samples=n, time=time.time() - t_start, reward_per_step=host[0])
        for i, name in enumerate(envs):
            st = host[5 + i * len(S):5 + (i + 1) * len(S)]
            ex = max(st[S["exited"]], 1.0)
            tag = name[:3]
            row.update({f"{tag}_success": st[S["success"]] / ex,
                        f"{tag}_coll_per_1k": 1000 * st[S["collisions"]] / ex,
                        f"{tag}_viol_per_1k": 1000 * st[S["violations"]] / ex,
                        f"{tag}_lc_per_veh": st[S["lane_changes"]] / ex,
                        f"{tag}_delay_s": st[S["delay"]] / ex})
        row.update(pg=host[1], vf=host[2], entropy=host[3], clipfrac=host[4])
        history.append(row)
        if it % log_every == 0 or it == 1:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if it % 25 == 0 or it == tc.iterations:
            save(net, tc, out_dir, history)
    return net, history


def save(net, tc, out_dir, history):
    torch.save({"state_dict": net.state_dict(), "hidden": tc.hidden,
                "config": dataclasses.asdict(tc)}, os.path.join(out_dir, "policy.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    net = ActorCritic(ck["hidden"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--envs", type=int, default=512, help="freeway segments")
    p.add_argument("--ix-envs", type=int, default=256, help="intersections (0 = freeway only)")
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--minibatch", type=int, default=TrainConfig.minibatch)
    p.add_argument("--reward", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override RewardConfig fields, e.g. --reward missed_route=-4")
    p.add_argument("--init", default=None, help="start from this policy.pt (fine-tuning)")
    p.add_argument("--out", default="runs/lc_ppo")
    a = p.parse_args()
    tc = TrainConfig(iterations=a.iterations, rollout_steps=a.rollout, seed=a.seed, lr=a.lr,
                     minibatch=a.minibatch)
    tc.env.num_envs = a.envs
    tc.ix.num_envs = a.ix_envs
    tc.use_ix = a.ix_envs > 0
    for kv in a.reward:
        k, v = kv.split("=")
        setattr(tc.reward, k, float(v))
    train(tc, a.out, init=a.init)
