# env_chess_boss_sf.py
import os
import random
from typing import Tuple, Dict, Any, Optional

import numpy as np
import chess
import chess.engine


class ChessBossEnvSF:
    """
    白棋：RL agent
    黑棋：Stockfish Boss（依 level 調難度）

    Step 2A（可達 / 可學 / 可累積）的核心：
    - 使用 Stockfish eval 形成「軟勝利 / 軟失敗」事件（Soft Win/Lose）
    - 連續 N 次 eval 超過閾值才觸發（避免單點噪聲）
    - 觸發後「重開局」，done 通常為 False（持續任務）

    Demo 闖關規則（你指定）：
    - 從 Level 1 開始
    - Win 升級到 Level 2 / 3
    - Lose、Draw 不降級
    - ✅ Level 3 通關後（win）就停止遊戲（done=True, event="clear_all"）
      - terminal win: 一定會通關結束
      - soft win: 只有 soft_win_levelup=True 且升到 Level3/已在Level3 才會通關結束
    """

    def __init__(
        self,
        lock_level1: bool = True,
        engine_path: Optional[str] = None,

        # --- 關卡 ---
        max_level: int = 3,

        # --- Demo 模式：Win 升，Lose/Draw 不降 ---
        demo_mode: bool = False,
        soft_win_levelup: bool = False,   # 若 True：soft win 也會升級（展示時可開）

        # --- 對局終止保護（避免太久）---
        max_plies: int = 240,

        # --- Reward（終局勝負）---
        win_reward: float = 1.2,
        lose_reward: float = -1.0,
        draw_reward: float = -0.6,

        # --- shaping ---
        material_coeff: float = 0.02,
        check_bonus: float = 0.02,
        check_penalty: float = -0.02,
        step_penalty: float = -0.00005,

        # --- Step 2A：Soft Win/Lose（以 eval 觸發）---
        use_soft_result: bool = True,
        eval_depth: int = 6,
        eval_interval_plies: int = 10,
        soft_win_eval: float = +2.5,
        soft_lose_eval: float = -2.5,
        soft_streak: int = 4,
        soft_win_reward: float = 0.35,
        soft_lose_reward: float = -0.45,
        soft_draw_reward: float = -0.45,

        # --- Stockfish 速度設定 ---
        hash_mb: int = 32,
        threads: int = 1,
    ):
        self.level = 1
        self.max_level = int(max_level)
        self.lock_level1 = bool(lock_level1)

        self.demo_mode = bool(demo_mode)
        self.soft_win_levelup = bool(soft_win_levelup)

        self.board = chess.Board()
        self.state_shape = (8, 8, 12)

        self.max_plies = int(max_plies)

        # rewards
        self.win_reward = float(win_reward)
        self.lose_reward = float(lose_reward)
        self.draw_reward = float(draw_reward)

        self.soft_win_reward = float(soft_win_reward)
        self.soft_lose_reward = float(soft_lose_reward)
        self.soft_draw_reward = float(soft_draw_reward)

        # shaping
        self.material_coeff = float(material_coeff)
        self.check_bonus = float(check_bonus)
        self.check_penalty = float(check_penalty)
        self.step_penalty = float(step_penalty)

        # soft result configs
        self.use_soft_result = bool(use_soft_result)
        self.eval_depth = int(eval_depth)
        self.eval_interval_plies = int(eval_interval_plies)
        self.soft_win_eval = float(soft_win_eval)
        self.soft_lose_eval = float(soft_lose_eval)
        self.soft_streak = int(soft_streak)

        # curriculum step（可選）
        self._global_step: int = 0

        # streak state
        self._adv_streak = 0
        self._dis_streak = 0

        # ---- 終局指標（給 eval_curve_sf / GUI 用）----
        self.last_terminal_material: Optional[float] = None
        self.last_terminal_plies: Optional[int] = None
        self.last_terminal_fen: Optional[str] = None   # 終局瞬間的 FEN
        self.last_event: str = ""                      # 最後事件字串

        # engine
        self.engine: Optional[chess.engine.SimpleEngine] = None

        if engine_path is None:
            engine_path = os.path.join(os.path.dirname(__file__), "stockfish.exe")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"[ChessBossEnvSF] 找不到 Stockfish 執行檔：{engine_path}\n"
                f"請把 stockfish 可執行檔放到該位置，或初始化時指定 engine_path。\n"
            )

        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        try:
            self.engine.configure({"Hash": int(hash_mb), "Threads": int(threads)})
        except Exception:
            pass

    # ----------------- clean up -----------------

    def close(self):
        if getattr(self, "engine", None) is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----------------- external helpers -----------------

    def set_global_step(self, step: int):
        self._global_step = int(step)

    def reset(self) -> np.ndarray:
        self.board = chess.Board()
        self._adv_streak = 0
        self._dis_streak = 0
        return self._board_to_state()

    def legal_action_ids(self):
        ids = []
        for mv in self.board.legal_moves:
            base = mv.from_square * 64 + mv.to_square
            if mv.promotion is None:
                ids.append(base * 5 + 0)
            else:
                promo_idx = self._promo_piece_to_idx(mv.promotion)  # 1..4
                ids.append(base * 5 + promo_idx)
        return ids

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # forced draw：避免一盤永遠下不完
        if self._plies() >= self.max_plies:
            r, done, info = self._handle_forced_draw("max_plies")
            return self._board_to_state(), r, done, info

        material_before = self._evaluate_material_for_white(self.board)

        # ============ 1) 白棋走 ============
        move = self._action_to_move_20480(action)

        if (self.board.turn != chess.WHITE) or (move not in self.board.legal_moves):
            r, done, info = self._handle_illegal_move()
            return self._board_to_state(), r, done, info

        self.board.push(move)

        white_gives_check = self.board.is_check() and (self.board.turn == chess.BLACK)

        if self.board.is_game_over():
            result = self.board.result()
            r, done, info = self._handle_result(result)
            return self._board_to_state(), r, done, info

        # ============ 2) 黑棋（Stockfish）走 ============
        boss_move = self._boss_move_stockfish()
        if boss_move is not None:
            self.board.push(boss_move)

        boss_gives_check = self.board.is_check() and (self.board.turn == chess.WHITE)

        if self.board.is_game_over():
            result = self.board.result()
            r, done, info = self._handle_result(result)
            return self._board_to_state(), r, done, info

        # ============ 3) shaping reward ============
        material_after = self._evaluate_material_for_white(self.board)
        delta_mat = material_after - material_before

        reward = self.step_penalty + self.material_coeff * float(delta_mat)
        if white_gives_check:
            reward += self.check_bonus
        if boss_gives_check:
            reward += self.check_penalty

        # ============ 4) Step 2A：soft win/lose（eval streak） ============
        if self.use_soft_result and self._should_eval_now():
            ev = self._stockfish_eval_pawns(depth=self.eval_depth)  # 白方 POV，單位：兵
            if ev is not None:
                if ev >= self.soft_win_eval:
                    self._adv_streak += 1
                    self._dis_streak = 0
                elif ev <= self.soft_lose_eval:
                    self._dis_streak += 1
                    self._adv_streak = 0
                else:
                    self._adv_streak = 0
                    self._dis_streak = 0

                # ---- soft win ----
                if self._adv_streak >= self.soft_streak:
                    self._record_terminal_snapshot(event="win_adv_eval")
                    self._adv_streak = 0
                    self._dis_streak = 0

                    # ✅ soft win 是否升級（展示可開）
                    if self.soft_win_levelup:
                        self._level_up_if_possible()

                        # ✅ 若到達最高關，直接通關停止（done=True）
                        cleared = self._clear_if_level3()
                        if cleared is not None:
                            r2, done2, info2 = cleared
                            return self._board_to_state(), r2, done2, info2

                    self.board = chess.Board()
                    return self._board_to_state(), self.soft_win_reward, False, {
                        "event": "win_adv_eval",
                        "eval_pawns": float(ev),
                        "level": int(self.level),
                    }

                # ---- soft lose ----
                if self._dis_streak >= self.soft_streak:
                    self._record_terminal_snapshot(event="lose_adv_eval")
                    self._adv_streak = 0
                    self._dis_streak = 0

                    self.board = chess.Board()
                    return self._board_to_state(), self.soft_lose_reward, False, {
                        "event": "lose_adv_eval",
                        "eval_pawns": float(ev),
                        "level": int(self.level),
                    }

        if self._plies() >= self.max_plies:
            r, done, info = self._handle_forced_draw("max_plies")
            return self._board_to_state(), r, done, info

        return self._board_to_state(), float(reward), False, {"event": ""}

    # ----------------- Stockfish boss -----------------

    def _boss_move_stockfish(self) -> Optional[chess.Move]:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves or self.board.turn != chess.BLACK:
            return None
        if self.engine is None:
            return random.choice(legal_moves)

        if self.level == 1:
            skill = 0
            move_time = 0.01
            rand_eps = 0.25
        elif self.level == 2:
            skill = 5
            move_time = 0.03
            rand_eps = 0.10
        else:
            skill = 10
            move_time = 0.08
            rand_eps = 0.05

        # curriculum：Level1 早期更亂、後期更穩
        if self.level == 1:
            s = max(0, self._global_step)
            if s <= 800_000:
                rand_eps = 0.45 + (0.25 - 0.45) * (s / 800_000.0)
            else:
                rand_eps = 0.25

        if random.random() < rand_eps:
            return random.choice(legal_moves)

        try:
            self.engine.configure({"Skill Level": int(skill)})
        except Exception:
            pass

        try:
            res = self.engine.play(self.board, limit=chess.engine.Limit(time=float(move_time)))
            mv = res.move
        except Exception:
            return random.choice(legal_moves)

        if mv is None or mv not in legal_moves:
            return random.choice(legal_moves)
        return mv

    # ----------------- soft eval helpers -----------------

    def _should_eval_now(self) -> bool:
        if self.eval_interval_plies <= 0:
            return True
        return (self._plies() % self.eval_interval_plies) == 0

    def _stockfish_eval_pawns(self, depth: int = 8) -> Optional[float]:
        if self.engine is None:
            return None
        try:
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=int(depth)))
            score = info["score"].pov(chess.WHITE)
            cp = score.score(mate_score=10000)
            if cp is None:
                return None
            return float(cp) / 100.0
        except Exception:
            return None

    # ----------------- demo level-up helpers -----------------

    def _level_up_if_possible(self):
        if self.level < self.max_level:
            self.level += 1

    def _clear_if_level3(self) -> Optional[Tuple[float, bool, Dict[str, Any]]]:
        """
        ✅ 若已在最高關（Level3）且剛剛發生「應視為通關的 win」，
        回傳 (reward, done, info)。否則回傳 None。
        """
        if self.level >= self.max_level:
            self.board = chess.Board()
            return float(self.win_reward), True, {"event": "clear_all", "level": int(self.level)}
        return None

    # ----------------- terminal snapshot -----------------

    def _record_terminal_snapshot(self, event: str):
        self.last_terminal_material = float(self._evaluate_material_for_white(self.board))
        self.last_terminal_plies = int(self._plies())
        self.last_terminal_fen = self.board.fen()
        self.last_event = str(event)

    # ----------------- result handling -----------------

    def _handle_forced_draw(self, reason: str) -> Tuple[float, bool, Dict[str, Any]]:
        self._record_terminal_snapshot(event=f"draw_forced_{reason}")
        r = self.soft_draw_reward
        self.board = chess.Board()
        return float(r), False, {"event": f"draw_forced_{reason}", "level": int(self.level)}

    def _handle_illegal_move(self) -> Tuple[float, bool, Dict[str, Any]]:
        self._record_terminal_snapshot(event="illegal")
        r = self.lose_reward
        self.board = chess.Board()
        return float(r), False, {"event": "illegal_level1", "level": int(self.level)}

    def _handle_result(self, result: str) -> Tuple[float, bool, Dict[str, Any]]:
        # 先存終局盤面
        if result == "1-0":
            self._record_terminal_snapshot(event="win_terminal")

            if self.demo_mode:
                if self.level < self.max_level:
                    self.level += 1
                    self.board = chess.Board()
                    return float(self.win_reward), False, {
                        "event": f"win_level_up_to_{self.level}",
                        "level": int(self.level),
                    }
                else:
                    self.board = chess.Board()
                    return float(self.win_reward), True, {"event": "clear_all", "level": int(self.level)}

            self.board = chess.Board()
            return float(self.win_reward), False, {"event": "win_terminal", "level": int(self.level)}

        if result == "0-1":
            self._record_terminal_snapshot(event="lose_terminal")
            if self.demo_mode:
                self.board = chess.Board()
                return float(self.lose_reward), False, {
                    "event": f"lose_stay_level_{self.level}",
                    "level": int(self.level),
                }
            self.board = chess.Board()
            return float(self.lose_reward), False, {"event": "lose_terminal", "level": int(self.level)}

        # draw
        self._record_terminal_snapshot(event="draw_terminal")
        if self.demo_mode:
            self.board = chess.Board()
            return float(self.draw_reward), False, {
                "event": f"draw_stay_level_{self.level}",
                "level": int(self.level),
            }
        self.board = chess.Board()
        return float(self.draw_reward), False, {"event": "draw_terminal", "level": int(self.level)}

    # ----------------- action decoding (20480) -----------------

    def _action_to_move_20480(self, action: int) -> chess.Move:
        base = int(action) // 5
        promo_idx = int(action) % 5
        from_sq = base // 64
        to_sq = base % 64

        m = chess.Move(from_sq, to_sq)
        if m in self.board.legal_moves:
            return m

        promo_piece = self._promo_idx_to_piece(promo_idx)
        pm = chess.Move(from_sq, to_sq, promotion=promo_piece)
        if pm in self.board.legal_moves:
            return pm

        for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            pm2 = chess.Move(from_sq, to_sq, promotion=p)
            if pm2 in self.board.legal_moves:
                return pm2

        return m

    @staticmethod
    def _promo_idx_to_piece(promo_idx: int) -> int:
        if promo_idx == 2:
            return chess.ROOK
        if promo_idx == 3:
            return chess.BISHOP
        if promo_idx == 4:
            return chess.KNIGHT
        return chess.QUEEN

    @staticmethod
    def _promo_piece_to_idx(piece_type: int) -> int:
        if piece_type == chess.QUEEN:
            return 1
        if piece_type == chess.ROOK:
            return 2
        if piece_type == chess.BISHOP:
            return 3
        if piece_type == chess.KNIGHT:
            return 4
        return 1

    # ----------------- state / material -----------------

    def _plies(self) -> int:
        return (self.board.fullmove_number - 1) * 2 + (0 if self.board.turn == chess.WHITE else 1)

    @staticmethod
    def _piece_value(piece_type: int) -> int:
        if piece_type == chess.PAWN:
            return 1
        if piece_type in (chess.KNIGHT, chess.BISHOP):
            return 3
        if piece_type == chess.ROOK:
            return 5
        if piece_type == chess.QUEEN:
            return 9
        return 0

    def _evaluate_material_for_white(self, board: chess.Board) -> float:
        score = 0
        for piece in board.piece_map().values():
            v = self._piece_value(piece.piece_type)
            score += v if piece.color == chess.WHITE else -v
        return float(score)

    def _board_to_state(self) -> np.ndarray:
        state = np.zeros(self.state_shape, dtype=np.float32)
        piece_type_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            base = 0 if piece.color == chess.WHITE else 6
            ch = base + piece_type_to_idx[piece.piece_type]
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            state[r, f, ch] = 1.0

        return state
