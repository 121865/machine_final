import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
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


class MLPlay:
    """
    1P 推論（穩健版，風格與 2P 一致）
    - 低溫抽樣（TEMPERATURE=0.12）
    - deadband 防抖（DEADBAND=1.0）
    - 防貼牆（WALL_MARGIN=5）
    - 球遠離才回中（RETURN_CENTER_ZONE=0.70）
    - 強制追擊很保守（FORCE_CHASE_DIST=10，只在差很大時才推一把）
    """

    WIDTH = 200.0
    HEIGHT = 500.0
    PADDLE_W = 40.0
    BALL_SIZE = 10.0
    MAX_V = 40.0

    # ---- 與 2P 對齊的穩健參數 ----
    DO_SAMPLE = True
    TEMPERATURE = 0.12

    DEADBAND = 1.0
    WALL_MARGIN = 5.0

    RETURN_CENTER_ZONE = 0.70
    FORCE_CHASE_DIST = 10.0

    MODEL_PATH = os.getenv("PPO_MODEL_1P", "ppo_pong_1P.pt")

    A2CMD: Dict[int, str] = {
        0: "MOVE_LEFT",
        1: "MOVE_RIGHT",
        2: "NONE",
        3: "SERVE_TO_LEFT",
        4: "SERVE_TO_RIGHT",
    }

    def __init__(self, ai_name: str, *args, **kwargs):
        self.ai_name = ai_name
        self.device = torch.device("cpu")

        self.model = ActorCritic(obs_dim=8, n_actions=5).to(self.device)

        if not os.path.isfile(self.MODEL_PATH):
            raise FileNotFoundError(f"[MLPlay-1P] 找不到模型檔：{self.MODEL_PATH}")

        sd = torch.load(self.MODEL_PATH, map_location=self.device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        self.model.load_state_dict(sd)
        self.model.eval()

        self.rng = np.random.default_rng()

        print(f"[MLPlay-1P] Loading PPO model from: {self.MODEL_PATH}")
        print("[MLPlay-1P] Model loaded successfully.")
        print("1P init OK")

    def reset(self):
        pass

    def _is_served(self, scene_info: dict) -> bool:
        if "ball_served" in scene_info:
            return bool(scene_info["ball_served"])
        bs = scene_info.get("ball_speed", [0, 0])
        return not (bs[0] == 0 and bs[1] == 0)

    def _make_obs(self, scene_info: dict) -> np.ndarray:
        bx, by = scene_info["ball"]
        vx, vy = scene_info.get("ball_speed", [0, 0])

        p1x = float(scene_info["platform_1P"][0])
        p2x = float(scene_info["platform_2P"][0])

        block = scene_info.get("blocker", None)
        blkx = float(block[0]) if block is not None else 0.0

        served = 1.0 if self._is_served(scene_info) else 0.0

        bx_n = (float(bx) / self.WIDTH) * 2.0 - 1.0
        by_n = (float(by) / self.HEIGHT) * 2.0 - 1.0
        vx_n = float(np.clip(float(vx) / self.MAX_V, -1.0, 1.0))
        vy_n = float(np.clip(float(vy) / self.MAX_V, -1.0, 1.0))
        p1_n = (p1x / (self.WIDTH - self.PADDLE_W)) * 2.0 - 1.0
        p2_n = (p2x / (self.WIDTH - self.PADDLE_W)) * 2.0 - 1.0
        blk_n = (blkx / (self.WIDTH - self.BALL_SIZE)) * 2.0 - 1.0

        return np.array([bx_n, by_n, vx_n, vy_n, p1_n, p2_n, blk_n, served], dtype=np.float32)

    def _select_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            logits = logits.squeeze(0).cpu().numpy()

        if not self.DO_SAMPLE:
            return int(np.argmax(logits))

        t = max(float(self.TEMPERATURE), 1e-6)
        z = logits / t
        z = z - np.max(z)
        p = np.exp(z)
        p = p / np.sum(p)
        return int(self.rng.choice(len(p), p=p))

    def _postprocess_cmd(self, scene_info: dict, cmd: str) -> str:
        px = float(scene_info["platform_1P"][0])
        bx, by = scene_info["ball"]
        bx, by = float(bx), float(by)

        ball_cx = bx + self.BALL_SIZE * 0.5
        paddle_cx = px + self.PADDLE_W * 0.5

        # (1) deadband 防抖
        if cmd in ("MOVE_LEFT", "MOVE_RIGHT") and abs(ball_cx - paddle_cx) <= self.DEADBAND:
            cmd = "NONE"

        # (2) 牆邊保護
        left_limit = self.WALL_MARGIN
        right_limit = self.WIDTH - self.PADDLE_W - self.WALL_MARGIN
        if cmd == "MOVE_LEFT" and px <= left_limit:
            cmd = "NONE"
        elif cmd == "MOVE_RIGHT" and px >= right_limit:
            cmd = "NONE"

        # 球速
        vx, vy = scene_info.get("ball_speed", [0, 0])
        vx, vy = float(vx), float(vy)

        toward_me = (vy > 0)
        my_zone = (by >= self.HEIGHT * self.RETURN_CENTER_ZONE)

        # (3) 球遠離才回中（穩健）
        if (not toward_me) and (not my_zone):
            center_x = (self.WIDTH - self.PADDLE_W) * 0.5
            if px < center_x - 1.0:
                return "MOVE_RIGHT"
            if px > center_x + 1.0:
                return "MOVE_LEFT"
            return "NONE"

        # (4) 強制追擊（保守，只在差很大才介入）
        if toward_me and abs(ball_cx - paddle_cx) >= self.FORCE_CHASE_DIST:
            cmd = "MOVE_RIGHT" if ball_cx > paddle_cx else "MOVE_LEFT"
            if cmd == "MOVE_LEFT" and px <= left_limit:
                cmd = "NONE"
            elif cmd == "MOVE_RIGHT" and px >= right_limit:
                cmd = "NONE"

        return cmd

    def update(self, scene_info: dict, keyboard_info=None) -> str:
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        served = self._is_served(scene_info)

        # 未發球：球在下半場才由 1P 發球；否則等 2P 發
        if not served:
            bx, by = scene_info["ball"]
            if float(by) >= self.HEIGHT * 0.5:
                return "SERVE_TO_LEFT" if self.rng.random() < 0.5 else "SERVE_TO_RIGHT"
            return "NONE"

        obs = self._make_obs(scene_info)
        act = self._select_action(obs)
        cmd = self.A2CMD.get(act, "NONE")
        return self._postprocess_cmd(scene_info, cmd)
