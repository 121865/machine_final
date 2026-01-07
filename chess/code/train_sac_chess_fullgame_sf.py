# train_sac_chess_fullgame_sf.py
import os
import re
import time
import random
from typing import Optional

from env_chess_boss_sf import ChessBossEnvSF
from sac_agent import SacAgent


def find_latest_checkpoint(models_dir: str) -> Optional[str]:
    if not os.path.isdir(models_dir):
        return None

    pattern = re.compile(r"^sac_chess_step(\d+)_actor\.pth$")
    best_step = -1
    best_prefix = None

    for fname in os.listdir(models_dir):
        m = pattern.match(fname)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_prefix = os.path.join(models_dir, f"sac_chess_step{step}")

    return best_prefix


def sample_level(p1: float, p2: float, p3: float) -> int:
    """回傳 1/2/3，依照給定機率抽樣。"""
    s = p1 + p2 + p3
    if s <= 0:
        return 1
    r = random.random() * s
    if r < p1:
        return 1
    if r < p1 + p2:
        return 2
    return 3


def is_new_game_event(event: str) -> bool:
    """
    判斷這一步是否「結束一盤並重開局」。
    你的 env 設計：terminal/soft/forced draw/illegal 都會 reset board。
    """
    if not event:
        return False
    if event.startswith("win_") or event.startswith("lose_"):
        return True
    if event.startswith("draw_"):
        return True
    if "illegal" in event:
        return True
    if event == "clear_all":
        return True
    return False


def run_training(
    num_steps: int = 200_000,
    updates_per_step: int = 1,
    log_interval: int = 1000,
    save_interval: int = 10_000,
    resume_prefix: Optional[str] = None,
    auto_resume: bool = True,

    # ===== 混合採樣機率（可調）=====
    p_level1: float = 0.60,
    p_level2: float = 0.30,
    p_level3: float = 0.10,
    seed: int = 0,
):
    random.seed(seed)

    # ====== 資料夾（你現在 v3）======
    models_dir = "models_sf_v3"
    os.makedirs(models_dir, exist_ok=True)

    # ====== loss log ======
    loss_log_path = "loss_log_sf_v3.csv"
    need_header = (not os.path.exists(loss_log_path)) or (os.path.getsize(loss_log_path) == 0)
    loss_f = open(loss_log_path, "a", buffering=1, encoding="utf-8")
    if need_header:
        loss_f.write("step,actor_loss,critic1_loss,critic2_loss,alpha_loss,alpha_value\n")

    # ====== 自動接續 ======
    if resume_prefix is None and auto_resume:
        latest = find_latest_checkpoint(models_dir)
        if latest is not None:
            resume_prefix = latest
            print(f"[AUTO] Found latest checkpoint: {resume_prefix}")
        else:
            print(f"[AUTO] No checkpoint found in {models_dir}, start from scratch.")

    # ====== Stockfish 路徑 ======
    engine_path = os.path.join(os.path.dirname(__file__), "stockfish.exe")

    # ====== ✅ Level-aware soft thresholds（你可在這裡調整）=====
    # 越高關卡門檻越嚴格：避免 Level1 的鬆條件把 policy 拉偏
    soft_win_eval_by_level = {
        1: 2.0,
        2: 3.0,
        3: 4.5,
    }
    soft_lose_eval_by_level = {
        1: -2.0,
        2: -3.0,
        3: -4.5,
    }

    # ====== 建立環境（用你 Step2A env）======
    env = ChessBossEnvSF(
        lock_level1=True,
        engine_path=engine_path,

        # Step 2A：軟勝利/軟失敗設定
        use_soft_result=True,
        eval_depth=8,
        eval_interval_plies=4,

        # ✅ 改成 Level-aware（不再使用 soft_win_eval / soft_lose_eval）
        soft_win_eval_by_level=soft_win_eval_by_level,
        soft_lose_eval_by_level=soft_lose_eval_by_level,

        soft_streak=3,
        soft_win_reward=0.55,
        soft_lose_reward=-0.75,
        soft_draw_reward=-0.75,

        max_plies=240,

        # 訓練不要開 demo_mode（避免 win 就升級造成分佈漂移）
        demo_mode=False,
        soft_win_levelup=False,
    )

    # ====== 建立 agent（20480 action space）======
    agent = SacAgent(
        state_shape=env.state_shape,
        action_dim=20480,
    )

    # ====== 載入 checkpoint ======
    global_step_offset = 0
    if resume_prefix is not None:
        print(f"[INFO] Loading checkpoint: {resume_prefix}")
        agent.load(resume_prefix)

        m = re.search(r"step(\d+)", resume_prefix)
        if m:
            global_step_offset = int(m.group(1))
            print(f"[INFO] Resuming from global step {global_step_offset}")
        else:
            print("[WARN] Cannot parse step number from prefix, global steps will start from 0.")

    # ====== 混合採樣：先抽第一盤的 Level ======
    env.level = sample_level(p_level1, p_level2, p_level3)

    # ====== init ======
    state = env.reset()
    legal_actions = env.legal_action_ids()

    total_reward = 0.0
    win_count = 0
    lose_count = 0
    draw_count = 0
    illegal_count = 0

    # 額外：看每個 log_interval 抽到多少局各 level
    lvl1_games = 0
    lvl2_games = 0
    lvl3_games = 0
    # 開局算一盤
    if env.level == 1:
        lvl1_games += 1
    elif env.level == 2:
        lvl2_games += 1
    else:
        lvl3_games += 1

    # loss 累積（log_interval 內平均）
    sum_actor_loss = 0.0
    sum_critic1_loss = 0.0
    sum_critic2_loss = 0.0
    sum_alpha_loss = 0.0
    sum_alpha_value = 0.0
    loss_update_count = 0
    update_calls = 0

    t0 = time.time()

    for local_t in range(1, num_steps + 1):
        global_t = global_step_offset + local_t

        # curriculum step -> env（你 boss 的 rand_eps 會用到）
        env.set_global_step(global_t)

        # 1) 選動作
        action = agent.select_action(state, legal_actions, eval_mode=False)

        # 2) env step
        next_state, reward, done, info = env.step(action)
        next_legal_actions = env.legal_action_ids()

        # 3) store
        agent.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            legal_actions=legal_actions,
            next_legal_actions=next_legal_actions,
        )

        total_reward += reward

        # 4) 統計事件
        event = info.get("event", "")

        if event in ("win_terminal", "win_adv_eval") or event.startswith("win_"):
            win_count += 1
        elif event in ("lose_terminal", "lose_adv_eval") or event.startswith("lose_"):
            lose_count += 1
        elif event.startswith("draw_") or ("draw" in event):
            draw_count += 1
        elif "illegal" in event:
            illegal_count += 1
            lose_count += 1

        # 4.5) ✅ 混合採樣：只要這一步結束一盤，就抽下一盤 Level
        if is_new_game_event(event):
            new_lv = sample_level(p_level1, p_level2, p_level3)
            env.level = new_lv
            if new_lv == 1:
                lvl1_games += 1
            elif new_lv == 2:
                lvl2_games += 1
            else:
                lvl3_games += 1

        # 5) update
        out = agent.update(updates_per_step=updates_per_step)
        if out is not None:
            update_calls += 1

        if agent.last_actor_loss is not None:
            sum_actor_loss += agent.last_actor_loss
            sum_critic1_loss += agent.last_critic1_loss
            sum_critic2_loss += agent.last_critic2_loss
            sum_alpha_loss += agent.last_alpha_loss
            if agent.last_alpha_value is not None:
                sum_alpha_value += agent.last_alpha_value
            loss_update_count += 1

        # 6) 推進狀態
        if done:
            state = env.reset()
            legal_actions = env.legal_action_ids()
        else:
            state = next_state
            legal_actions = next_legal_actions

        # 7) log
        if global_t % log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = log_interval / max(elapsed, 1e-6)
            avg_reward = total_reward / log_interval
            buf_size = len(agent.buffer)

            if loss_update_count > 0:
                avg_actor_loss = sum_actor_loss / loss_update_count
                avg_c1_loss = sum_critic1_loss / loss_update_count
                avg_c2_loss = sum_critic2_loss / loss_update_count
                avg_alpha_loss = sum_alpha_loss / loss_update_count
                avg_alpha_value = sum_alpha_value / loss_update_count
            else:
                avg_actor_loss = float("nan")
                avg_c1_loss = float("nan")
                avg_c2_loss = float("nan")
                avg_alpha_loss = float("nan")
                avg_alpha_value = float("nan")

            print(
                f"Step {global_t} | AvgReward: {avg_reward:.3f} | "
                f"Wins: {win_count} | Loses: {lose_count} | Draws: {draw_count} | Illegals: {illegal_count} | "
                f"CurLevel: {env.level} | "
                f"Games(L1/L2/L3): {lvl1_games}/{lvl2_games}/{lvl3_games} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"Buf: {buf_size} | Upd: {update_calls} | "
                f"ActorLoss: {avg_actor_loss:.4f} | Critic1Loss: {avg_c1_loss:.4f} | Critic2Loss: {avg_c2_loss:.4f} | "
                f"AlphaLoss: {avg_alpha_loss:.4f} | Alpha: {avg_alpha_value:.4f}"
            )

            # 寫 CSV（避免 NaN）
            if not any(x != x for x in [avg_actor_loss, avg_c1_loss, avg_c2_loss, avg_alpha_loss, avg_alpha_value]):
                loss_f.write(
                    f"{global_t},"
                    f"{avg_actor_loss:.6f},"
                    f"{avg_c1_loss:.6f},"
                    f"{avg_c2_loss:.6f},"
                    f"{avg_alpha_loss:.6f},"
                    f"{avg_alpha_value:.6f}\n"
                )

            # reset interval stats
            total_reward = 0.0
            win_count = lose_count = draw_count = illegal_count = 0
            lvl1_games = lvl2_games = lvl3_games = 0

            sum_actor_loss = sum_critic1_loss = sum_critic2_loss = sum_alpha_loss = sum_alpha_value = 0.0
            loss_update_count = 0
            update_calls = 0
            t0 = time.time()

        # 8) save
        if global_t % save_interval == 0:
            prefix = os.path.join(models_dir, f"sac_chess_step{global_t}")
            agent.save(prefix)
            print(f"[INFO] Saved checkpoint at step {global_t} → {prefix}_*.pth")

    loss_f.close()
    env.close()
    print("Training finished.")


if __name__ == "__main__":
    run_training(
        num_steps=200_000,
        updates_per_step=1,
        log_interval=1000,
        save_interval=10_000,
        resume_prefix=None,
        auto_resume=True,

        # 混合採樣（預設 60/30/10）
        p_level1=0.60,
        p_level2=0.30,
        p_level3=0.10,
        seed=0,
    )
