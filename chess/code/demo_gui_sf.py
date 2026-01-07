# demo_gui_sf.py
import os
import re
import tkinter as tk
from tkinter import ttk
from typing import Optional

import chess

from env_chess_boss_sf import ChessBossEnvSF
from sac_agent import SacAgent


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def find_latest_checkpoint(models_dir: str) -> Optional[str]:
    if not os.path.isdir(models_dir):
        return None
    best_step = -1
    best_prefix = None
    pat = re.compile(r"^sac_chess_step(\d+)_actor\.pth$")
    for fn in os.listdir(models_dir):
        m = pat.match(fn)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_prefix = os.path.join(models_dir, f"sac_chess_step{step}")
    return best_prefix


class ChessDemoGUI:
    def __init__(
        self,
        models_dir: str = "models_sf_v3",
        engine_path: Optional[str] = None,
        step_delay_ms: int = 200,         # 自動播放速度
        max_white_moves_per_game: int = 400,  # 單局最多白棋步數（避免拖太久）
    ):
        self.models_dir = models_dir
        self.step_delay_ms = int(step_delay_ms)
        self.max_white_moves_per_game = int(max_white_moves_per_game)

        if engine_path is None:
            engine_path = os.path.join(os.path.dirname(__file__), "stockfish.exe")

        # ---- env：展示用
        # demo_mode=True 讓「真終局 win」會升級；Level3 再 win -> clear_all
        # soft_win_levelup=True 讓 soft win 也可以升級（若你不想，可改 False）
        self.env = ChessBossEnvSF(
            lock_level1=False,
            engine_path=engine_path,
            demo_mode=True,
            soft_win_levelup=True,
            use_soft_result=True,
        )
        self.env.level = 1

        # ---- agent
        self.agent = SacAgent(state_shape=self.env.state_shape, action_dim=20480)
        ckpt = find_latest_checkpoint(models_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {models_dir}")
        self.agent.load(ckpt)

        # ---- demo state
        self.state = self.env.reset()
        self.legal = self.env.legal_action_ids()
        self.white_moves_in_game = 0

        # 進度：已通關幾關（0~3）
        self.cleared_count = 0
        self.last_result_text = "-"

        self.running = False
        self.cleared = False

        # 用來偵測升級，並「停住讓你看到通關畫面」
        self.prev_level = int(self.env.level)
        self.pause_for_levelup = False

        # ---- UI
        self.root = tk.Tk()
        self.root.title("SAC Chess Boss Demo (Level 1 -> 3)")

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Board frame
        board_frame = ttk.Frame(main)
        board_frame.grid(row=0, column=0, padx=(0, 12), sticky="n")

        self.squares = [[None] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                lbl = tk.Label(
                    board_frame,
                    width=4, height=2,
                    font=("Consolas", 20),
                    relief="solid",
                    bd=1,
                )
                lbl.grid(row=r, column=c, sticky="nsew")
                self.squares[r][c] = lbl

        # Right panel
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nw")

        self.lbl_level = ttk.Label(right, text="Level: 1", font=("Segoe UI", 14, "bold"))
        self.lbl_level.grid(row=0, column=0, sticky="w")

        self.lbl_event = ttk.Label(right, text="Event: ", font=("Segoe UI", 11))
        self.lbl_event.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.lbl_last = ttk.Label(right, text="Last Result: -", font=("Segoe UI", 11, "bold"))
        self.lbl_last.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.lbl_progress = ttk.Label(right, text="Progress: Cleared 0/3", font=("Segoe UI", 11))
        self.lbl_progress.grid(row=3, column=0, sticky="w", pady=(6, 0))

        self.lbl_hint = ttk.Label(
            right,
            text="說明：從 Level1 開始\nWin 升級，Lose/Draw 不降\n通關 Level3 後顯示成功並停止",
            font=("Segoe UI", 10),
        )
        self.lbl_hint.grid(row=4, column=0, sticky="w", pady=(10, 0))

        btns = ttk.Frame(right)
        btns.grid(row=5, column=0, sticky="w", pady=(14, 0))

        self.btn_start = ttk.Button(btns, text="Start / Resume", command=self.start)
        self.btn_start.grid(row=0, column=0, padx=(0, 8))

        self.btn_pause = ttk.Button(btns, text="Pause", command=self.pause)
        self.btn_pause.grid(row=0, column=1, padx=(0, 8))

        self.btn_step = ttk.Button(btns, text="Step Once", command=self.step_once)
        self.btn_step.grid(row=0, column=2)

        self.banner = ttk.Label(right, text="", font=("Segoe UI", 14, "bold"), foreground="green")
        self.banner.grid(row=6, column=0, sticky="w", pady=(18, 0))

        # Initial draw
        self.render_board_from_env()
        self.update_labels(event_text="")

    # ---------------- UI helpers ----------------

    def fen_to_board(self, fen: str) -> chess.Board:
        b = chess.Board()
        b.set_fen(fen)
        return b

    def render_board(self, board: chess.Board):
        # tkinter row0 是上方，所以用 rank 8->1 視角
        for ui_r in range(8):
            rank = 7 - ui_r
            for file_ in range(8):
                sq = chess.square(file_, rank)
                piece = board.piece_at(sq)
                bg = "#EEEED2" if (ui_r + file_) % 2 == 0 else "#769656"
                ch = ""
                if piece:
                    ch = UNICODE_PIECES[piece.symbol()]
                lbl = self.squares[ui_r][file_]
                lbl.configure(text=ch, bg=bg, fg="black")

    def render_board_from_env(self):
        self.render_board(self.env.board)

    def update_labels(self, event_text: str):
        self.lbl_level.configure(text=f"Level: {self.env.level}")
        self.lbl_event.configure(text=f"Event: {event_text}")
        self.lbl_last.configure(text=f"Last Result: {self.last_result_text}")
        self.lbl_progress.configure(text=f"Progress: Cleared {self.cleared_count}/3")

    # ---------------- control ----------------

    def start(self):
        if self.cleared:
            return
        self.running = True
        self.loop()

    def pause(self):
        self.running = False

    def step_once(self):
        if self.cleared:
            return
        self.running = False
        self._do_one_env_step()

    # ---------------- core step ----------------

    def _set_last_result_by_event(self, event: str):
        if not event:
            return
        e = event.lower()
        if "clear_all" in e:
            self.last_result_text = "CLEAR"
        elif e.startswith("win") or "win_" in e:
            self.last_result_text = "WIN"
        elif e.startswith("lose") or "lose_" in e or "illegal" in e:
            self.last_result_text = "LOSE"
        elif e.startswith("draw") or "draw_" in e:
            self.last_result_text = "DRAW"

    def _is_new_game_event(self, event: str) -> bool:
        if not event:
            return False
        if event.startswith(("win_", "lose_", "draw_")):
            return True
        if "illegal" in event:
            return True
        if event == "clear_all":
            return True
        return False

    def _do_one_env_step(self):
        # 若已超過單局上限，直接重開（避免卡住）
        if self.white_moves_in_game >= self.max_white_moves_per_game:
            self.white_moves_in_game = 0
            self.state = self.env.reset()
            self.legal = self.env.legal_action_ids()
            self.render_board_from_env()
            self.update_labels(event_text="force_reset_by_demo_limit")
            return

        # 1) 先記下 step 前 level（用來判斷是否升級）
        before_level = int(self.env.level)

        # 2) agent 走 deterministic（展示更穩）
        action = self.agent.select_action(self.state, self.legal, eval_mode=True)

        # 3) env step
        next_state, reward, done, info = self.env.step(action)
        event = info.get("event", "")

        self.state = next_state
        self.legal = self.env.legal_action_ids()
        self.white_moves_in_game += 1

        after_level = int(self.env.level)
        level_up_happened = (after_level > before_level)

        # 4) 如果有事件：顯示「終局瞬間盤面」
        if event:
            self._set_last_result_by_event(event)

            # 顯示終局盤面（如果 env 有存 FEN）
            if getattr(self.env, "last_terminal_fen", None):
                term_fen = self.env.last_terminal_fen
                if term_fen:
                    self.render_board(self.fen_to_board(term_fen))
                else:
                    self.render_board_from_env()
            else:
                self.render_board_from_env()

            # 5) 升級事件：一定要停住，避免下一步立刻覆蓋畫面
            if level_up_happened:
                # before_level 被通關了（例如 1->2 表示通關 1）
                cleared_level = before_level
                self.cleared_count = max(self.cleared_count, cleared_level)
                self.banner.configure(text=f"✅ 通關 Level {cleared_level}！按 Start 進入 Level {after_level}")
                self.update_labels(event_text=event)

                self.running = False
                self.white_moves_in_game = 0
                return

            # 6) 最終通關（Level3 win 後的 clear_all）
            if event == "clear_all":
                self.cleared_count = 3
                self.banner.configure(text="🎉 闖關成功：已通關 Level 3！")
                self.update_labels(event_text=event)

                self.cleared = True
                self.running = False
                self.white_moves_in_game = 0
                return

            # 7) 其他事件（lose/draw/soft lose/forced draw 等）：不升級，只更新顯示
            self.banner.configure(text="")
            self.update_labels(event_text=event)
            if self._is_new_game_event(event):
                self.white_moves_in_game = 0
            return

        # 沒事件：正常走子顯示
        self.banner.configure(text="")
        self.render_board_from_env()
        self.update_labels(event_text="")

    def loop(self):
        if not self.running or self.cleared:
            return
        self._do_one_env_step()
        self.root.after(self.step_delay_ms, self.loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = ChessDemoGUI(
        models_dir="models_sf_v3",
        step_delay_ms=200,
        max_white_moves_per_game=400,
    )
    gui.run()
