# sac_agent.py
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================
#   Network
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
#   Replay item
# =========================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    legal_actions: List[int]
    next_legal_actions: List[int]


# =========================
#   SAC Agent (Discrete)
# =========================

class SacAgent:
    """
    Discrete SAC with action masking (legal moves).

    - Actor outputs logits over action_dim
    - Critics output Q(s, a) for all a (vector length action_dim)
    - Target uses masked soft value on next_state:
        V(s') = sum_{a in legal(s')} pi(a|s') [ min(Q1,Q2) - alpha * log pi(a|s') ]
    """

    def __init__(
        self,
        state_shape=(8, 8, 12),
        action_dim: int = 20480,
        device=None,
        gamma: float = 0.99,
        alpha: float = 0.002,             # 固定 alpha（穩）
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-5,
        buffer_size: int = 200_000,
        batch_size: int = 64,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        hidden_dim: int = 256,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.state_shape = state_shape
        self.state_dim = int(np.prod(state_shape))
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.max_grad_norm = float(max_grad_norm)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Actor: logits over actions
        self.actor = MLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)

        # Critics: Q(s, ·)
        self.critic1 = MLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic2 = MLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)

        # Target critics
        self.critic1_target = MLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic2_target = MLP(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=float(critic_lr))
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=float(critic_lr))

        # Replay buffer
        self.buffer: Deque[Transition] = deque(maxlen=int(buffer_size))
        self.training_steps = 0

        # log 用
        self.last_actor_loss: Optional[float] = None
        self.last_critic1_loss: Optional[float] = None
        self.last_critic2_loss: Optional[float] = None
        self.last_alpha_loss: Optional[float] = 0.0       # 固定 alpha -> alpha_loss 固定 0
        self.last_alpha_value: Optional[float] = self.alpha

    # -------------------------
    #   Public API
    # -------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_actions: List[int],
        next_legal_actions: List[int],
    ):
        self.buffer.append(
            Transition(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
                legal_actions=list(legal_actions),
                next_legal_actions=list(next_legal_actions),
            )
        )

    @torch.no_grad()
    def select_action(self, state: np.ndarray, legal_actions: List[int], eval_mode: bool = False) -> int:
        """
        - eval_mode=False：依 pi(a|s) 取樣（探索）
        - eval_mode=True：取 argmax pi(a|s)（較 deterministic）
        """
        legal = self._sanitize_legal_actions(legal_actions)
        if len(legal) == 0:
            # 保底：回傳 0（理論上 env 不應該讓這發生）
            return 0

        s = torch.from_numpy(state.astype(np.float32)).view(1, -1).to(self.device)
        logits = self.actor(s).squeeze(0)  # [A]

        masked_logits = self._mask_logits_1d(logits, legal)

        # 若 masked_logits 全是 -inf / nan，直接 random 合法步
        if torch.isnan(masked_logits).any():
            return int(random.choice(legal))

        if eval_mode:
            a = int(torch.argmax(masked_logits).item())
            # 再保險一次：如果 argmax 落在不合法（理論上不會），退回 random
            if a not in set(legal):
                return int(random.choice(legal))
            return a

        probs = torch.softmax(masked_logits, dim=-1)

        # 防呆：避免 nan 或 sum=0
        if torch.isnan(probs).any() or (probs.sum() <= 0):
            return int(random.choice(legal))

        dist = torch.distributions.Categorical(probs=probs)
        a = int(dist.sample().item())

        if a not in set(legal):
            # 極少數數值問題保底
            return int(random.choice(legal))
        return a

    def update(self, updates_per_step: int = 1):
        """
        做 updates_per_step 次更新。
        若 buffer 不足，直接略過，不會產生 nan。
        """
        # 只有在真的 update 後才會填 loss
        self.last_actor_loss = None
        self.last_critic1_loss = None
        self.last_critic2_loss = None
        self.last_alpha_loss = 0.0
        self.last_alpha_value = self.alpha

        for _ in range(int(updates_per_step)):
            if len(self.buffer) < self.batch_size:
                return
            self._update_once()

    def save(self, path_prefix: str):
        path_prefix = str(path_prefix)
        d = os.path.dirname(path_prefix)
        if d:
            os.makedirs(d, exist_ok=True)

        torch.save(self.actor.state_dict(),   path_prefix + "_actor.pth")
        torch.save(self.critic1.state_dict(), path_prefix + "_critic1.pth")
        torch.save(self.critic2.state_dict(), path_prefix + "_critic2.pth")
        torch.save(self.critic1_target.state_dict(), path_prefix + "_critic1_target.pth")
        torch.save(self.critic2_target.state_dict(), path_prefix + "_critic2_target.pth")

    def load(self, path_prefix: str):
        path_prefix = str(path_prefix)

        def _load_sd(path: str):
            try:
                return torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # 舊版 torch 不支援 weights_only
                return torch.load(path, map_location=self.device)

        self.actor.load_state_dict(_load_sd(path_prefix + "_actor.pth"))
        self.critic1.load_state_dict(_load_sd(path_prefix + "_critic1.pth"))
        self.critic2.load_state_dict(_load_sd(path_prefix + "_critic2.pth"))

        c1t = path_prefix + "_critic1_target.pth"
        c2t = path_prefix + "_critic2_target.pth"
        if os.path.exists(c1t) and os.path.exists(c2t):
            self.critic1_target.load_state_dict(_load_sd(c1t))
            self.critic2_target.load_state_dict(_load_sd(c2t))
        else:
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic2_target.load_state_dict(self.critic2.state_dict())

    # -------------------------
    #   Internal
    # -------------------------

    def _update_once(self):
        self.training_steps += 1
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.from_numpy(np.stack([b.state for b in batch]).astype(np.float32)).view(self.batch_size, -1).to(self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device).view(self.batch_size, 1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).view(self.batch_size, 1)
        next_states = torch.from_numpy(np.stack([b.next_state for b in batch]).astype(np.float32)).view(self.batch_size, -1).to(self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).view(self.batch_size, 1)

        # 重要：把 batch 的 legal_actions 做 sanitize，避免 out-of-bounds
        legal_batch = [self._sanitize_legal_actions(b.legal_actions) for b in batch]
        next_legal_batch = [self._sanitize_legal_actions(b.next_legal_actions) for b in batch]

        # ====== Critic target ======
        with torch.no_grad():
            next_logits = self.actor(next_states)  # [B, A]
            next_log_probs, next_probs = self._masked_logprob_and_prob(next_logits, next_legal_batch)

            q1_next = self.critic1_target(next_states)  # [B, A]
            q2_next = self.critic2_target(next_states)  # [B, A]
            q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)  # [B,1]
            target_q = rewards + (1.0 - dones) * self.gamma * v_next  # [B,1]

        # ====== Critic loss ======
        # actions 若超界，gather 會炸；所以這裡也做一次 clamp（保底）
        actions_safe = actions.clamp(min=0, max=self.action_dim - 1)

        q1 = self.critic1(states).gather(1, actions_safe)  # [B,1]
        q2 = self.critic2(states).gather(1, actions_safe)  # [B,1]

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad(set_to_none=True)
        critic1_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_opt.step()

        # ====== Actor loss ======
        logits = self.actor(states)  # [B, A]
        log_probs, probs = self._masked_logprob_and_prob(logits, legal_batch)

        with torch.no_grad():
            q1_pi = self.critic1(states)
            q2_pi = self.critic2(states)
            q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (probs * (self.alpha * log_probs - q_pi)).sum(dim=1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # ====== Soft update target critics ======
        self._soft_update(self.critic1_target, self.critic1, self.tau)
        self._soft_update(self.critic2_target, self.critic2, self.tau)

        # ====== logging ======
        self.last_actor_loss = float(actor_loss.detach().cpu().item())
        self.last_critic1_loss = float(critic1_loss.detach().cpu().item())
        self.last_critic2_loss = float(critic2_loss.detach().cpu().item())
        self.last_alpha_loss = 0.0
        self.last_alpha_value = float(self.alpha)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    # -------------------------
    #   Masking helpers (safe)
    # -------------------------

    def _sanitize_legal_actions(self, legal_actions: List[int]) -> List[int]:
        """
        避免 CUDA index out-of-bounds：
        - 移除 <0 或 >= action_dim
        - 去重
        """
        if not legal_actions:
            return []
        out = []
        seen = set()
        for a in legal_actions:
            try:
                ai = int(a)
            except Exception:
                continue
            if ai < 0 or ai >= self.action_dim:
                continue
            if ai in seen:
                continue
            seen.add(ai)
            out.append(ai)
        return out

    def _mask_logits_1d(self, logits_1d: torch.Tensor, legal_actions: List[int]) -> torch.Tensor:
        """
        logits_1d: [A]
        將非法動作設成 -1e9（近似 -inf），避免 softmax 有機率
        """
        if len(legal_actions) == 0:
            # 全非法：回傳全 -1e9（讓上層退回 random/保底）
            return torch.full_like(logits_1d, -1e9)

        masked = torch.full_like(logits_1d, -1e9)
        idx = torch.tensor(legal_actions, dtype=torch.long, device=logits_1d.device)
        masked.index_fill_(0, idx, 0.0)
        return logits_1d + masked

    def _masked_logprob_and_prob(
        self,
        logits: torch.Tensor,                 # [B, A]
        legal_actions_batch: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        回傳：
          log_probs: [B, A]（非法動作的 log_prob 接近 -inf）
          probs    : [B, A]（非法動作 prob=0）
        """
        B, A = logits.shape
        device = logits.device

        # 建 mask: 合法=0, 非法=-1e9
        mask = torch.full((B, A), -1e9, device=device, dtype=logits.dtype)

        for i, legal in enumerate(legal_actions_batch):
            if not legal:
                # 若這列沒有合法（理論上不該發生），退回「不遮罩」
                mask[i, :] = 0.0
            else:
                idx = torch.tensor(legal, dtype=torch.long, device=device)
                mask[i, idx] = 0.0

        masked_logits = logits + mask
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = torch.exp(log_probs)

        # 防呆：若出現 nan，改用 row normalize
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            probs = probs / row_sum
            log_probs = torch.log(probs.clamp(min=1e-12))

        return log_probs, probs
