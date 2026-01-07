import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ===========================
# 1) PAIA 風格 Pong self-play 環境（1P/2P 同時）
# ===========================
class PaiaPongEnvSelfPlay:
    """
    以 PAIA pingpong 規則為藍本的簡化環境（self-play 版）

    - 畫面大小：200 x 500（px）
    - 板子：40 x 10，1P 在下方 (x, 420)，2P 在上方 (x, 70)，每步移動 5 px
    - 球：10 x 10，初速 (±7, ±7)，每 100 frame 速度 +1
    - HARD 模式會有障礙物（blocker）在 y=240 左右來回

    行動空間（離散 5 個動作）：0:L 1:R 2:NONE 3:SERVE_L 4:SERVE_R

    reward:
        +1/-1 得分
        +hit_bonus：成功擊球
        +align_coef：球朝向自己時，板子越對準球中心越加分（小獎勵）
    """

    def __init__(self, difficulty="NORMAL"):
        self.width = 200
        self.height = 500

        self.paddle_w = 40
        self.paddle_h = 10
        self.paddle_speed = 5

        self.ball_size = 10
        self.init_speed = 7
        self.speed_increase_interval = 100
        self.max_speed_for_draw = 40

        self.p1_y = 420
        self.p2_y = 70

        self.difficulty = difficulty
        self.blocker_w = 30
        self.blocker_h = 20
        self.blocker_speed = 5

        self.max_steps = 2000

        # ===== Reward shaping（讓模型更容易學會移動/對齊/回擊）=====
        # 注意：係數要小於得分(+1/-1)很多，避免學歪
        self.hit_bonus = 0.05      # 成功擊到球的額外獎勵
        self.align_coef = 0.01     # 球朝向自己時，對齊球中心的小獎勵

        self.steps = 0
        self._speedup_counter = 0

        self._last_move_1 = 0
        self._last_move_2 = 0
        self._blocker_dir = 1

        # server: 1 表示 1P 發球；2 表示 2P 發球
        self.server = 1

        self.state = None

    @property
    def observation_space_shape(self):
        # 1P/2P 都是 8 維
        return (8,)

    @property
    def n_actions(self):
        return 5

    def reset(self):
        self.steps = 0
        self._speedup_counter = 0

        p1_x = 80.0
        p2_x = 80.0

        served = False

        # 球黏在發球方板子旁
        if self.server == 1:
            ball_x = p1_x + self.paddle_w / 2 - self.ball_size / 2
            ball_y = self.p1_y - self.ball_size
        else:
            ball_x = p2_x + self.paddle_w / 2 - self.ball_size / 2
            ball_y = self.p2_y + self.paddle_h

        vel_x = 0.0
        vel_y = 0.0

        # HARD 模式 blocker
        if self.difficulty == "HARD":
            blk_x = np.random.choice(np.arange(0, self.width - self.blocker_w + 1, 20))
            blk_y = 240.0
        else:
            blk_x = -1.0
            blk_y = -1.0

        self._last_move_1 = 0
        self._last_move_2 = 0
        self._blocker_dir = np.random.choice([-1, 1])

        self.state = np.array(
            [ball_x, ball_y, vel_x, vel_y, p1_x, p2_x, blk_x, blk_y, float(served)],
            dtype=np.float32,
        )

        obs1 = self._build_obs_1p()
        obs2 = self._build_obs_2p()
        return obs1, obs2

    def _build_obs_1p(self):
        (ball_x, ball_y, vel_x, vel_y, p1_x, p2_x, blk_x, blk_y, served_flag) = self.state
        width = float(self.width)
        height = float(self.height)
        max_v = float(self.max_speed_for_draw)

        bx = float(ball_x) / width
        by = float(ball_y) / height
        vx = float(vel_x) / max_v
        vy = float(vel_y) / max_v
        p1 = float(p1_x) / width
        p2 = float(p2_x) / width
        blk = float(blk_x) / width if blk_x >= 0 else 0.0
        served = float(served_flag)

        return np.array([bx, by, vx, vy, p1, p2, blk, served], dtype=np.float32)

    def _build_obs_2p(self):
        (ball_x, ball_y, vel_x, vel_y, p1_x, p2_x, blk_x, blk_y, served_flag) = self.state
        width = float(self.width)
        height = float(self.height)
        max_v = float(self.max_speed_for_draw)

        bx = float(ball_x) / width
        by = (height - float(ball_y) - self.ball_size) / height
        vx = float(vel_x) / max_v
        vy = -float(vel_y) / max_v
        self_p = float(p2_x) / width
        opp_p = float(p1_x) / width
        blk = float(blk_x) / width if blk_x >= 0 else 0.0
        served = float(served_flag)

        return np.array([bx, by, vx, vy, self_p, opp_p, blk, served], dtype=np.float32)

    def step(self, action_1: int, action_2: int):
        self.steps += 1

        (ball_x, ball_y, vel_x, vel_y, p1_x, p2_x, blk_x, blk_y, served_flag) = self.state
        served = bool(served_flag)

        r1 = 0.0
        r2 = 0.0
        done = False

        hit1 = False
        hit2 = False

        # paddle 1
        move1 = 0
        if action_1 == 0:
            p1_x -= self.paddle_speed
            move1 = -1
        elif action_1 == 1:
            p1_x += self.paddle_speed
            move1 = +1
        self._last_move_1 = move1
        p1_x = np.clip(p1_x, 0.0, self.width - self.paddle_w)

        # paddle 2
        move2 = 0
        if action_2 == 0:
            p2_x -= self.paddle_speed
            move2 = -1
        elif action_2 == 1:
            p2_x += self.paddle_speed
            move2 = +1
        self._last_move_2 = move2
        p2_x = np.clip(p2_x, 0.0, self.width - self.paddle_w)

        # blocker
        if self.difficulty == "HARD":
            blk_x += self._blocker_dir * self.blocker_speed
            if blk_x <= 0 or blk_x >= self.width - self.blocker_w:
                blk_x = np.clip(blk_x, 0.0, self.width - self.blocker_w)
                self._blocker_dir *= -1
        else:
            blk_x = -1.0
            blk_y = -1.0

        # serve
        if not served:
            if self.server == 1:
                ball_x = p1_x + self.paddle_w / 2 - self.ball_size / 2
                ball_y = self.p1_y - self.ball_size
                if action_1 in (3, 4):
                    served = True
                    dir_x = -1 if action_1 == 3 else +1
                    vel_x = dir_x * self.init_speed
                    vel_y = -self.init_speed
                    self._speedup_counter = 0
            else:
                ball_x = p2_x + self.paddle_w / 2 - self.ball_size / 2
                ball_y = self.p2_y + self.paddle_h
                if action_2 in (3, 4):
                    served = True
                    dir_x = -1 if action_2 == 3 else +1
                    vel_x = dir_x * self.init_speed
                    vel_y = +self.init_speed
                    self._speedup_counter = 0

        # ball physics
        if served and not done:
            ball_x += vel_x
            ball_y += vel_y

            self._speedup_counter += 1
            if self._speedup_counter >= self.speed_increase_interval:
                self._speedup_counter = 0
                speed = float(np.sqrt(vel_x**2 + vel_y**2))
                if speed > 1e-6:
                    new_speed = speed + 1.0
                    scale = new_speed / speed
                    vel_x *= scale
                    vel_y *= scale

            # walls
            if ball_x <= 0:
                ball_x = 0
                vel_x = abs(vel_x)
            elif ball_x + self.ball_size >= self.width:
                ball_x = self.width - self.ball_size
                vel_x = -abs(vel_x)

            # collide with 1P paddle (bottom)
            if (
                ball_y + self.ball_size >= self.p1_y
                and ball_y <= self.p1_y + self.paddle_h
                and ball_x + self.ball_size >= p1_x
                and ball_x <= p1_x + self.paddle_w
                and vel_y > 0
            ):
                ball_y = self.p1_y - self.ball_size
                hit1 = True
                if self._last_move_1 == 0:
                    vel_y = -abs(vel_y)
                else:
                    if np.sign(self._last_move_1) == np.sign(vel_x if vel_x != 0 else self._last_move_1):
                        vel_x += 3 * np.sign(vel_x if vel_x != 0 else self._last_move_1)
                    else:
                        vel_x = -vel_x if vel_x != 0 else 3 * self._last_move_1
                    vel_y = -abs(vel_y)

            # collide with 2P paddle (top)
            if (
                ball_y <= self.p2_y + self.paddle_h
                and ball_y + self.ball_size >= self.p2_y
                and ball_x + self.ball_size >= p2_x
                and ball_x <= p2_x + self.paddle_w
                and vel_y < 0
            ):
                ball_y = self.p2_y + self.paddle_h
                hit2 = True
                if self._last_move_2 == 0:
                    vel_y = abs(vel_y)
                else:
                    if np.sign(self._last_move_2) == np.sign(vel_x if vel_x != 0 else self._last_move_2):
                        vel_x += 3 * np.sign(vel_x if vel_x != 0 else self._last_move_2)
                    else:
                        vel_x = -vel_x if vel_x != 0 else 3 * self._last_move_2
                    vel_y = abs(vel_y)

            # blocker collision
            if self.difficulty == "HARD" and blk_x >= 0:
                if (
                    ball_x + self.ball_size >= blk_x
                    and ball_x <= blk_x + self.blocker_w
                    and ball_y + self.ball_size >= blk_y
                    and ball_y <= blk_y + self.blocker_h
                ):
                    if vel_y > 0:
                        ball_y = blk_y - self.ball_size
                        vel_y = -abs(vel_y)
                    else:
                        ball_y = blk_y + self.blocker_h
                        vel_y = abs(vel_y)

            # ===== Reward shaping =====
            # 1) 擊球獎勵：鼓勵主動回擊
            if hit1:
                r1 += self.hit_bonus
            if hit2:
                r2 += self.hit_bonus

            # 2) 對齊獎勵：球朝向自己時，鼓勵板子對準球的 x
            ball_cx = float(ball_x + self.ball_size * 0.5)
            p1_cx = float(p1_x + self.paddle_w * 0.5)
            p2_cx = float(p2_x + self.paddle_w * 0.5)
            max_dist = self.width * 0.5  # 最大可能中心距離（近似）

            if vel_y > 0:  # 球往下 → 1P 準備接球
                align = 1.0 - abs(ball_cx - p1_cx) / max_dist
                r1 += self.align_coef * float(np.clip(align, 0.0, 1.0))
            elif vel_y < 0:  # 球往上 → 2P 準備接球
                align = 1.0 - abs(ball_cx - p2_cx) / max_dist
                r2 += self.align_coef * float(np.clip(align, 0.0, 1.0))

            # score
            if ball_y + self.ball_size >= self.height:
                r1 = -1.0
                r2 = +1.0
                done = True
                self.server = 2
            elif ball_y <= 0:
                r1 = +1.0
                r2 = -1.0
                done = True
                self.server = 1

            # draw
            speed_now = float(np.sqrt(vel_x ** 2 + vel_y ** 2))
            if speed_now > self.max_speed_for_draw and not done:
                done = True
                self.server = 2 if self.server == 1 else 1

        # max steps
        if self.steps >= self.max_steps and not done:
            done = True
            self.server = 2 if self.server == 1 else 1

        self.state = np.array(
            [ball_x, ball_y, vel_x, vel_y, p1_x, p2_x, blk_x, blk_y, float(served)],
            dtype=np.float32,
        )

        obs1 = self._build_obs_1p()
        obs2 = self._build_obs_2p()
        return obs1, obs2, r1, r2, done, {}


# ===========================
# 2) Actor-Critic 網路
# ===========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.net(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# ===========================
# 3) GAE
# ===========================
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_term = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_term = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_term - values[t]
        last_gae = delta + gamma * lam * next_non_term * last_gae
        adv[t] = last_gae

    returns = adv + values
    return returns, adv


# ===========================
# 4) Rollout + PPO update (共用)
# ===========================
def ppo_update(model, optimizer, obs_tensor, act_tensor, ret_tensor, adv_tensor, oldlog_tensor,
               clip_eps=0.15, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, epochs=5, batch_size=256):
    logs = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [], "approx_kl": []}
    idxs = np.arange(obs_tensor.shape[0])

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            end = start + batch_size
            batch = idxs[start:end]

            b_obs = obs_tensor[batch]
            b_act = act_tensor[batch]
            b_ret = ret_tensor[batch]
            b_adv = adv_tensor[batch]
            b_oldlog = oldlog_tensor[batch]

            logits, values = model(b_obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(b_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - b_oldlog)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values = values.squeeze(-1)
            value_loss = (b_ret - values).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            with torch.no_grad():
                approx_kl = (b_oldlog - logp).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            logs["policy_loss"].append(float(policy_loss.item()))
            logs["value_loss"].append(float(value_loss.item()))
            logs["entropy"].append(float(entropy.item()))
            logs["total_loss"].append(float(loss.item()))
            logs["approx_kl"].append(float(approx_kl.item()))

    return logs


# ===========================
# 5) Loss plot（可選）
# ===========================
def save_loss_plots(out_dir: Path, logs: dict, tag: str):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(key):
        if key not in logs or len(logs[key]) == 0:
            return
        plt.figure()
        plt.plot(logs[key])
        plt.title(f"{tag} {key}")
        plt.xlabel("update step")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_{key}.png")
        plt.close()

    for k in ["policy_loss", "value_loss", "entropy", "total_loss", "approx_kl"]:
        _plot_one(k)


# ===========================
# 6) 兩個模型 self-play 訓練 + logging
# ===========================
def train_selfplay_ppo(
    total_steps=200_000,
    rollout_steps=2048,       # ✅ 變大更穩
    epochs=5,                 # ✅ 減少每次過度更新
    batch_size=256,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.15,            # ✅ 更保守
    lr=1e-4,                  # ✅ self-play 更穩
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    difficulty="NORMAL",
    device=None,
    save_plots=True,
    out_dir="out_selfplay",
):
    env = PaiaPongEnvSelfPlay(difficulty=difficulty)
    obs_dim = env.observation_space_shape[0]
    n_actions = env.n_actions

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = ActorCritic(obs_dim, n_actions).to(device)
    model2 = ActorCritic(obs_dim, n_actions).to(device)

    opt1 = optim.Adam(model1.parameters(), lr=lr)
    opt2 = optim.Adam(model2.parameters(), lr=lr)

    # logs（把每次 update 的平均值記下來，方便畫圖）
    logs_1p = {k: [] for k in ["policy_loss", "value_loss", "entropy", "total_loss", "approx_kl"]}
    logs_2p = {k: [] for k in ["policy_loss", "value_loss", "entropy", "total_loss", "approx_kl"]}

    # rollout buffers
    obs1_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
    obs2_buf = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
    act1_buf = np.zeros((rollout_steps,), dtype=np.int64)
    act2_buf = np.zeros((rollout_steps,), dtype=np.int64)
    rew1_buf = np.zeros((rollout_steps,), dtype=np.float32)
    rew2_buf = np.zeros((rollout_steps,), dtype=np.float32)
    done_buf = np.zeros((rollout_steps,), dtype=np.float32)
    val1_buf = np.zeros((rollout_steps,), dtype=np.float32)
    val2_buf = np.zeros((rollout_steps,), dtype=np.float32)
    logp1_buf = np.zeros((rollout_steps,), dtype=np.float32)
    logp2_buf = np.zeros((rollout_steps,), dtype=np.float32)

    obs1, obs2 = env.reset()
    global_step = 0

    while global_step < total_steps:
        # collect rollout
        for t in range(rollout_steps):
            global_step += 1

            obs1_t = torch.from_numpy(obs1).float().unsqueeze(0).to(device)
            obs2_t = torch.from_numpy(obs2).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits1, value1 = model1(obs1_t)
                dist1 = Categorical(logits=logits1)
                a1 = dist1.sample()
                lp1 = dist1.log_prob(a1)

                logits2, value2 = model2(obs2_t)
                dist2 = Categorical(logits=logits2)
                a2 = dist2.sample()
                lp2 = dist2.log_prob(a2)

            next_obs1, next_obs2, r1, r2, done, _ = env.step(int(a1.item()), int(a2.item()))

            obs1_buf[t] = obs1
            obs2_buf[t] = obs2
            act1_buf[t] = int(a1.item())
            act2_buf[t] = int(a2.item())
            rew1_buf[t] = float(r1)
            rew2_buf[t] = float(r2)
            done_buf[t] = float(done)
            val1_buf[t] = float(value1.item())
            val2_buf[t] = float(value2.item())
            logp1_buf[t] = float(lp1.item())
            logp2_buf[t] = float(lp2.item())

            obs1, obs2 = next_obs1, next_obs2

            if done:
                obs1, obs2 = env.reset()

            if global_step >= total_steps:
                break

        # last value
        with torch.no_grad():
            obs1_t = torch.from_numpy(obs1).float().unsqueeze(0).to(device)
            _, last_v1 = model1(obs1_t)
            obs2_t = torch.from_numpy(obs2).float().unsqueeze(0).to(device)
            _, last_v2 = model2(obs2_t)

        ret1, adv1 = compute_gae(rew1_buf, val1_buf, done_buf, float(last_v1.item()), gamma=gamma, lam=lam)
        ret2, adv2 = compute_gae(rew2_buf, val2_buf, done_buf, float(last_v2.item()), gamma=gamma, lam=lam)

        # normalize adv
        adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)
        adv2 = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)

        # tensors
        obs1_tensor = torch.from_numpy(obs1_buf).to(device)
        obs2_tensor = torch.from_numpy(obs2_buf).to(device)
        act1_tensor = torch.from_numpy(act1_buf).to(device)
        act2_tensor = torch.from_numpy(act2_buf).to(device)
        ret1_tensor = torch.from_numpy(ret1).to(device)
        ret2_tensor = torch.from_numpy(ret2).to(device)
        adv1_tensor = torch.from_numpy(adv1).to(device)
        adv2_tensor = torch.from_numpy(adv2).to(device)
        oldlog1_tensor = torch.from_numpy(logp1_buf).to(device)
        oldlog2_tensor = torch.from_numpy(logp2_buf).to(device)

        # update 1P
        ulog1 = ppo_update(
            model1, opt1,
            obs1_tensor, act1_tensor, ret1_tensor, adv1_tensor, oldlog1_tensor,
            clip_eps=clip_eps, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, epochs=epochs, batch_size=batch_size
        )
        # update 2P
        ulog2 = ppo_update(
            model2, opt2,
            obs2_tensor, act2_tensor, ret2_tensor, adv2_tensor, oldlog2_tensor,
            clip_eps=clip_eps, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, epochs=epochs, batch_size=batch_size
        )

        # append mean logs for plotting
        for k in logs_1p.keys():
            logs_1p[k].append(float(np.mean(ulog1[k])) if len(ulog1[k]) else 0.0)
            logs_2p[k].append(float(np.mean(ulog2[k])) if len(ulog2[k]) else 0.0)

        print(f"[Update] step={global_step:7d} | "
              f"1P: pl={logs_1p['policy_loss'][-1]:.4f} vl={logs_1p['value_loss'][-1]:.4f} ent={logs_1p['entropy'][-1]:.4f} "
              f"| 2P: pl={logs_2p['policy_loss'][-1]:.4f} vl={logs_2p['value_loss'][-1]:.4f} ent={logs_2p['entropy'][-1]:.4f}")

    out_dir = Path(out_dir)
    if save_plots:
        save_loss_plots(out_dir, logs_1p, "1P")
        save_loss_plots(out_dir, logs_2p, "2P")
        print(f"[Saved] loss plots -> {out_dir.resolve()}")

    return model1, model2, logs_1p, logs_2p


if __name__ == "__main__":
    model1, model2, logs_1p, logs_2p = train_selfplay_ppo(
        total_steps=200_000,
        difficulty="NORMAL",  # 可改 "HARD"
        save_plots=True,
    )

    torch.save(model1.state_dict(), "ppo_pong_1P.pt")
    torch.save(model2.state_dict(), "ppo_pong_2P.pt")
    print("訓練完成，模型已儲存為 ppo_pong_1P.pt / ppo_pong_2P.pt")
