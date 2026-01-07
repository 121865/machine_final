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

    Demo 闖關規則（你指定）：
    - 從 Level 1 開始
    - Win 升級到 Level 2 / 3
    - Lose、Draw 不降級
    - ✅ Level 3 win 後停止遊戲（done=True, event="clear_all"）

    Soft Result（Step 2A）：
    - 用 Stockfish eval 形成 soft win/lose（連續 soft_streak 次達標才觸發）
    - soft win 可選擇是否升級（soft_win_levelup=True 時才升級）
    """

    def __init__(
        self,
        lock_level1: bool = True,                  # 保留參數：避免外部舊程式呼叫炸掉（內部不使用）
        engine_path: Optional[str] = None,

        # --- 關卡 ---
        max_level: int = 3,

        # --- Demo 模式 ---
        demo_mode: bool = False,
        soft_win_levelup: bool = False,            # soft win 是否也升級（展示可開）

        # --- 對局終止保護 ---
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

        # --- Soft Win/Lose（eval 觸發）---
        use_soft_result: bool = True,
        eval_depth: int = 6,
        eval_interval_plies: int = 10,
        soft_win_eval: float = +2.5,
        soft_lose_eval: float = -2.5,
        soft_streak: int = 4,
        soft_win_reward: float = 0.35,
        soft_lose_reward: float = -0.45,
        soft_draw_reward: float = -0.45,

        # --- Stockfish ---
        hash_mb: int = 32,
        threads: int = 1,
    ):
        # --- basic states ---
        self.level = 1
        self.max_level = int(max_level)

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

        # soft config
        self.use_soft_result = bool(use_soft_result)
        self.eval_depth = int(eval_depth)
        self.eval_interval_plies = int(eval_interval_plies)
        self.soft_win_eval = float(soft_win_eval)
        self.soft_lose_eval = float(soft_lose_eval)
        self.soft_streak = int(soft_streak)

        self._adv_streak = 0
        self._dis_streak = 0

        # ✅ 保留給 demo_gui 用（但已大幅簡化，只留 fen/event）
        self.last_terminal_fen: Optional[str] = None
        self.last_event: str = ""

        # ✅ 保留接口：訓練腳本可能會呼叫（即使我們不用 curriculum，也不會炸）
        self._global_step: int = 0

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

    # ----------------- lifecycle -----------------

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

    # ----------------- public helpers -----------------

    def set_global_step(self, step: int):
        """保留 API：外部訓練可能會呼叫。"""
        self._global_step = int(step)

    def reset(self) -> np.ndarray:
        self.board = chess.Board()
        self._adv_streak = 0
        self._dis_streak = 0
        # reset 也把上一局的終局資訊清掉（避免 GUI 誤顯示）
        self.last_terminal_fen = None
        self.last_event = ""
        return self._board_to_state()

    def legal_action_ids(self):
        """
        action space = 20480 = 4096 * 5
          base = from*64 + to  (0..4095)
          promo_idx = 0..4
          action = base*5 + promo_idx

        非升變步：promo_idx 只能算 0（其他會變 illegal）
        升變步：promo_idx 0 表示「不指定」→ 預設升后（Q）
        """
        ids = []
        for mv in self.board.legal_moves:
            base = mv.from_square * 64 + mv.to_square
            if mv.promotion is None:
                ids.append(base * 5 + 0)
            else:
                ids.append(base * 5 + self._promo_piece_to_idx(mv.promotion))
        return ids

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # 0) forced draw（避免一局無限拖）
        if self._plies() >= self.max_plies:
            r, done, info = self._end_game(f"draw_forced_max_plies", self.soft_draw_reward, False)
            return self._board_to_state(), r, done, info

        material_before = self._evaluate_material_for_white(self.board)

        # 1) White move
        move = self._action_to_move_20480(action)
        if (self.board.turn != chess.WHITE) or (move not in self.board.legal_moves):
            r, done, info = self._end_game("illegal_level1", self.lose_reward, False)
            return self._board_to_state(), r, done, info

        self.board.push(move)
        white_gives_check = self.board.is_check() and (self.board.turn == chess.BLACK)

        if self.board.is_game_over():
            r, done, info = self._handle_terminal_result(self.board.result())
            return self._board_to_state(), r, done, info

        # 2) Black (Stockfish) move
        boss_move = self._boss_move_stockfish()
        if boss_move is not None:
            self.board.push(boss_move)

        boss_gives_check = self.board.is_check() and (self.board.turn == chess.WHITE)

        if self.board.is_game_over():
            r, done, info = self._handle_terminal_result(self.board.result())
            return self._board_to_state(), r, done, info

        # 3) shaping reward
        material_after = self._evaluate_material_for_white(self.board)
        delta_mat = material_after - material_before

        reward = self.step_penalty + self.material_coeff * float(delta_mat)
        if white_gives_check:
            reward += self.check_bonus
        if boss_gives_check:
            reward += self.check_penalty

        # 4) soft win/lose (eval streak)
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
                    self._adv_streak = 0
                    self._dis_streak = 0

                    # 展示：soft win 也可升級
                    if self.soft_win_levelup:
                        self._level_up_if_possible()

                        # 若升到最高關，直接通關
                        if self.level >= self.max_level:
                            r2, done2, info2 = self._end_game("clear_all", self.win_reward, True)
                            return self._board_to_state(), r2, done2, info2

                    # soft win 只是事件（通常 done=False）
                    r2, done2, info2 = self._end_game("win_adv_eval", self.soft_win_reward, False, extra={"eval_pawns": float(ev)})
                    return self._board_to_state(), r2, done2, info2

                # ---- soft lose ----
                if self._dis_streak >= self.soft_streak:
                    self._adv_streak = 0
                    self._dis_streak = 0
                    r2, done2, info2 = self._end_game("lose_adv_eval", self.soft_lose_reward, False, extra={"eval_pawns": float(ev)})
                    return self._board_to_state(), r2, done2, info2

        # 5) forced draw 再檢查一次
        if self._plies() >= self.max_plies:
            r, done, info = self._end_game("draw_forced_max_plies", self.soft_draw_reward, False)
            return self._board_to_state(), r, done, info

        return self._board_to_state(), float(reward), False, {"event": ""}

    # ----------------- end/terminal helpers -----------------

    def _end_game(
        self,
        event: str,
        reward: float,
        done: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        統一處理：記錄終局盤面(FEN) → reset board → 回傳 info
        """
        self.last_terminal_fen = self.board.fen()
        self.last_event = str(event)

        info: Dict[str, Any] = {"event": str(event), "level": int(self.level)}
        if extra:
            info.update(extra)

        # 結束一局就重開（clear_all 也重開，方便 GUI 顯示最後局面後停止）
        self.board = chess.Board()
        return float(reward), bool(done), info

    def _handle_terminal_result(self, result: str) -> Tuple[float, bool, Dict[str, Any]]:
        """
        result: "1-0", "0-1", "1/2-1/2"
        Demo 模式：Win 升級；Level3 win => clear_all(done=True)
        """
        # white win
        if result == "1-0":
            if self.demo_mode:
                # win: level up (or clear)
                if self.level < self.max_level:
                    self._level_up_if_possible()
                    return self._end_game(f"win_level_up_to_{self.level}", self.win_reward, False)
                else:
                    return self._end_game("clear_all", self.win_reward, True)

            # training / eval
            return self._end_game("win_terminal", self.win_reward, False)

        # white lose
        if result == "0-1":
            if self.demo_mode:
                return self._end_game(f"lose_stay_level_{self.level}", self.lose_reward, False)
            return self._end_game("lose_terminal", self.lose_reward, False)

        # draw
        if self.demo_mode:
            return self._end_game(f"draw_stay_level_{self.level}", self.draw_reward, False)
        return self._end_game("draw_terminal", self.draw_reward, False)

    def _level_up_if_possible(self):
        if self.level < self.max_level:
            self.level += 1

    # ----------------- Stockfish boss -----------------

    def _boss_move_stockfish(self) -> Optional[chess.Move]:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves or self.board.turn != chess.BLACK:
            return None
        if self.engine is None:
            return random.choice(legal_moves)

        # 三關
        if self.level == 1:
            skill, move_time, rand_eps = 0, 0.01, 0.25
        elif self.level == 2:
            skill, move_time, rand_eps = 5, 0.03, 0.10
        else:
            skill, move_time, rand_eps = 10, 0.08, 0.05

        # （保留簡易 curriculum：外部若呼叫 set_global_step 仍有效；不想要也可整段刪掉）
        if self.level == 1:
            s = max(0, self._global_step)
            if s <= 800_000:
                rand_eps = 0.45 + (0.25 - 0.45) * (s / 800_000.0)

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
        """回傳白方 POV 的 eval（單位 pawn）。"""
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

    # ----------------- action decoding (20480) -----------------

    def _action_to_move_20480(self, action: int) -> chess.Move:
        base = int(action) // 5
        promo_idx = int(action) % 5
        from_sq = base // 64
        to_sq = base % 64

        # no promotion
        m = chess.Move(from_sq, to_sq)
        if m in self.board.legal_moves:
            return m

        # promotion (0 -> queen default)
        promo_piece = self._promo_idx_to_piece(promo_idx)
        pm = chess.Move(from_sq, to_sq, promotion=promo_piece)
        if pm in self.board.legal_moves:
            return pm

        # fallback try Q/R/B/N
        for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            pm2 = chess.Move(from_sq, to_sq, promotion=p)
            if pm2 in self.board.legal_moves:
                return pm2

        return m  # illegal will be handled by caller

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
