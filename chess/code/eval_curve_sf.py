# eval_curve_sf.py
import os
import re
import glob
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import chess

from sac_agent import SacAgent
from env_chess_boss_sf import ChessBossEnvSF


# --------- 解析 checkpoint ---------

def list_checkpoints(models_dir: str) -> List[Tuple[int, str]]:
    """
    回傳 [(step, prefix), ...]
    prefix 形如: models_xxx/sac_chess_step7000000
    """
    pattern = os.path.join(models_dir, "sac_chess_step*_actor.pth")
    files = glob.glob(pattern)

    out = []
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"^sac_chess_step(\d+)_actor\.pth$", base)
        if not m:
            continue
        step = int(m.group(1))
        prefix = os.path.join(models_dir, f"sac_chess_step{step}")
        out.append((step, prefix))

    out.sort(key=lambda x: x[0])
    return out


# --------- 工具：子力差 / ply ---------

def piece_value(piece_type: int) -> int:
    # P=1, N/B=3, R=5, Q=9, K=0
    if piece_type == chess.PAWN:
        return 1
    if piece_type in (chess.KNIGHT, chess.BISHOP):
        return 3
    if piece_type == chess.ROOK:
        return 5
    if piece_type == chess.QUEEN:
        return 9
    return 0


def material_diff_white(board: chess.Board) -> float:
    # 白子力 - 黑子力
    score = 0
    for p in board.piece_map().values():
        v = piece_value(p.piece_type)
        score += v if p.color == chess.WHITE else -v
    return float(score)


def plies(board: chess.Board) -> int:
    # 半步數估計
    return (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)


# --------- 防呆：過濾 legal_actions 越界 ---------

def filter_legal_actions(legal_actions: List[int], action_dim: int) -> List[int]:
    if not legal_actions:
        return []
    ok = [a for a in legal_actions if 0 <= int(a) < action_dim]
    if len(ok) != len(legal_actions):
        bad = [int(a) for a in legal_actions if not (0 <= int(a) < action_dim)]
        print(f"[WARN] legal_actions out of range: action_dim={action_dim}, bad_examples={bad[:10]}")
    return ok


# --------- 評估單一 checkpoint ---------

def eval_one_checkpoint(
    env: ChessBossEnvSF,
    agent: SacAgent,
    games: int,
    eval_level: int,
    max_white_moves_per_game: int,
    action_dim: int,
) -> Dict[str, float]:
    """
    以目前 agent 權重，跑 games 盤（每盤最多 max_white_moves_per_game 個「白方 action」回合）
    回傳統計：
      win_rate, lose_rate, draw_rate,
      avg_terminal_material_diff, avg_terminal_plies
    """
    env.level = int(eval_level)

    win = lose = draw = illegal = 0
    terminal_mats: List[float] = []
    terminal_plies: List[int] = []

    for g in range(games):
        state = env.reset()

        # 每步都留快照，避免 env 在 event 時 reset 導致抓不到終局盤面
        board_snap: chess.Board = env.board.copy(stack=False)

        legal_actions = env.legal_action_ids()
        legal_actions = filter_legal_actions(legal_actions, action_dim)

        # 若真的空了（不正常），直接算輸
        if len(legal_actions) == 0:
            print("[WARN] legal_actions empty at start -> count as lose")
            lose += 1
            terminal_mats.append(material_diff_white(board_snap))
            terminal_plies.append(plies(board_snap))
            continue

        ended = False

        for _t in range(max_white_moves_per_game):
            # 評估用：eval_mode=True（不加探索）
            action = agent.select_action(state, legal_actions, eval_mode=True)

            next_state, reward, done, info = env.step(action)

            # 更新快照（這一步之後的盤面）
            board_snap = env.board.copy(stack=False)

            event = info.get("event", "")

            if event:
                # 用快照盤面算終局指標（不依賴 env 是否 reset）
                terminal_mats.append(material_diff_white(board_snap))
                terminal_plies.append(plies(board_snap))

                if event.startswith("win"):
                    win += 1
                elif event.startswith("lose"):
                    lose += 1
                elif event.startswith("draw"):
                    draw += 1
                elif event.startswith("illegal"):
                    illegal += 1
                    lose += 1
                else:
                    # 未知 event：保守當作 draw
                    draw += 1

                ended = True
                break

            state = next_state
            legal_actions = env.legal_action_ids()
            legal_actions = filter_legal_actions(legal_actions, action_dim)

            # 若 legal_actions 變空（通常是 mapping/版本不匹配），避免崩潰
            if len(legal_actions) == 0:
                print("[WARN] legal_actions empty mid-game -> count as lose")
                lose += 1
                terminal_mats.append(material_diff_white(board_snap))
                terminal_plies.append(plies(board_snap))
                ended = True
                break

            if done:
                # 你 lock_level1=True 通常 done=False；保險
                terminal_mats.append(material_diff_white(board_snap))
                terminal_plies.append(plies(board_snap))
                draw += 1
                ended = True
                break

        if not ended:
            # 超過 max_white_moves_per_game，算 draw（避免 eval 跑爆久）
            draw += 1
            terminal_mats.append(material_diff_white(board_snap))
            terminal_plies.append(plies(board_snap))

    total = max(win + lose + draw, 1)
    win_rate = win / total
    lose_rate = lose / total
    draw_rate = draw / total

    avg_mat = float(np.mean(terminal_mats)) if terminal_mats else float("nan")
    avg_pl = float(np.mean(terminal_plies)) if terminal_plies else float("nan")

    return {
        "win_rate": win_rate,
        "lose_rate": lose_rate,
        "draw_rate": draw_rate,
        "avg_terminal_material_diff": avg_mat,
        "avg_terminal_plies": avg_pl,
        "illegal_count": float(illegal),
    }


# --------- main ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models_sf_v3")
    parser.add_argument("--engine_path", type=str, default=None)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--max_white_moves_per_game", type=int, default=200)  # 白方回合上限（快一點）
    parser.add_argument("--save_prefix", type=str, default="eval_curve_sf_v3")
    parser.add_argument("--action_dim", type=int, default=20480)  # <<< models_sf_v3 用 20480
    args = parser.parse_args()

    ckpts = list_checkpoints(args.models_dir)
    if len(ckpts) == 0:
        print(f"[ERR] 找不到 checkpoint：{args.models_dir}/sac_chess_step*_actor.pth")
        return

    # env（Stockfish）
    env = ChessBossEnvSF(lock_level1=True, engine_path=args.engine_path)

    # agent（20480）
    agent = SacAgent(state_shape=env.state_shape, action_dim=args.action_dim)

    steps = []
    win_rates = []
    lose_rates = []
    draw_rates = []
    avg_mats = []
    avg_plies = []
    illegals = []

    for step, prefix in ckpts:
        print(f"[EVAL] step={step} prefix={prefix}")
        agent.load(prefix)

        stats = eval_one_checkpoint(
            env=env,
            agent=agent,
            games=args.games,
            eval_level=args.level,
            max_white_moves_per_game=args.max_white_moves_per_game,
            action_dim=args.action_dim,
        )

        steps.append(step)
        win_rates.append(stats["win_rate"])
        lose_rates.append(stats["lose_rate"])
        draw_rates.append(stats["draw_rate"])
        avg_mats.append(stats["avg_terminal_material_diff"])
        avg_plies.append(stats["avg_terminal_plies"])
        illegals.append(stats["illegal_count"])

        print(
            f"  win={stats['win_rate']:.2f}, lose={stats['lose_rate']:.2f}, draw={stats['draw_rate']:.2f}, "
            f"avg_terminal_mat={stats['avg_terminal_material_diff']:.2f}, avg_terminal_plies={stats['avg_terminal_plies']:.1f}, "
            f"illegals={int(stats['illegal_count'])}"
        )

    env.close()

    # --------- plot 1: W/L/D ---------
    plt.figure()
    plt.plot(steps, win_rates, label="Win rate")
    plt.plot(steps, lose_rates, label="Lose rate")
    plt.plot(steps, draw_rates, label="Draw rate")
    plt.xlabel("Training step (from checkpoint name)")
    plt.ylabel("Rate")
    plt.title(f"Eval vs Stockfish (each {args.games} games) - Level {args.level}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out1 = f"{args.save_prefix}_wld.png"
    plt.savefig(out1)
    print(f"[SAVE] {out1}")

    # --------- plot 2: Avg terminal material diff ---------
    plt.figure()
    plt.plot(steps, avg_mats, label="Avg terminal material (white-black)")
    plt.xlabel("Training step (from checkpoint name)")
    plt.ylabel("Material diff")
    plt.title(f"Avg Terminal Material vs Stockfish (each {args.games} games) - Level {args.level}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out2 = f"{args.save_prefix}_terminal_mat.png"
    plt.savefig(out2)
    print(f"[SAVE] {out2}")

    # --------- plot 3: Avg terminal plies ---------
    plt.figure()
    plt.plot(steps, avg_plies, label="Avg terminal plies")
    plt.xlabel("Training step (from checkpoint name)")
    plt.ylabel("Plies")
    plt.title(f"Avg Terminal Game Length vs Stockfish (each {args.games} games) - Level {args.level}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out3 = f"{args.save_prefix}_terminal_plies.png"
    plt.savefig(out3)
    print(f"[SAVE] {out3}")

    # --------- plot 4: Illegals (debug indicator) ---------
    plt.figure()
    plt.plot(steps, illegals, label="Illegal count (per eval batch)")
    plt.xlabel("Training step (from checkpoint name)")
    plt.ylabel("Count")
    plt.title(f"Illegals vs Stockfish (each {args.games} games) - Level {args.level}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out4 = f"{args.save_prefix}_illegals.png"
    plt.savefig(out4)
    print(f"[SAVE] {out4}")

    plt.show()


if __name__ == "__main__":
    main()
