Chess model  
===
介紹  
---
使用Soft Actor-Critic (SAC)訓練白棋來對抗Stockfish Boss，並以闖關制(Level 1~Level 3)來進行成果展示  
* 白棋 : RL Agent
* 黑棋 : Stockfish(依照Level調整難度)
* 玩法 : 從Level 1開始，Win則升級至Level 2，Lose / Draw不降級。通關Level 3即結束遊戲

架構圖
---
<img width="1398" height="416" alt="image" src="https://github.com/user-attachments/assets/997473f9-7428-4680-8ecf-3170b677b273" />  

Breakdown
---
<img width="1184" height="892" alt="image" src="https://github.com/user-attachments/assets/1b183228-3a21-4f24-abd2-d8ababf4c10a" />  

API  
---
### env_chess_boss_sf.py  
| API                       | Input                                                               | Output                        | Method                                 |
| ------------------------- | ------------------------------------------------------------------- | ----------------------------- | ---------------------------------------|
| `ChessBossEnvSF.__init__` | `demo_mode: bool`<br>`soft_win_levelup: bool`<br>`engine_path: str` | `env instance`                | 初始化環境、Stockfish、關卡規則          |
| `reset()`                 | –                                                                   | `state: np.ndarray`           | 重置棋盤並回傳初始狀態                   |
| `step(action)`            | `action: int (0~20479)`                                             | `(state, reward, done, info)` | 執行一個 white action，內含 black 回應   |
| `legal_action_ids()`      | –                                                                   | `List[int]`                   | 回傳目前合法 action（已編碼）            |
| `set_global_step()`       | `step: int`                                                         | –                             | 提供訓練進度給 Boss（curriculum 用）     |  

### sac_agent.py  
| API                 | Input                                           | Output           | Method                             |
| ------------------- | ----------------------------------------------- | ---------------- | -----------------------------------|
| `SacAgent.__init__` | `state_shape`<br>`action_dim=20480`             | `agent instance` | 初始化 SAC actor / critic           |
| `select_action()`   | `state`<br>`legal_actions`<br>`eval_mode: bool` | `action: int`    | 選擇合法 action（train / eval）     |
| `store()`           | `(s, a, r, s', done)`                           | –                | 儲存 transition 至 replay buffer    |
| `update()`          | –                                               | `loss dict`      | 執行 SAC 更新                       |
| `save()`            | `path prefix`                                   | –                | 儲存模型                            |
| `load()`            | `path prefix`                                   | –                | 載入模型                            |  
  
### train_sac_chess_fullgame_sf.py
| API                        | Input                             | Output           | Method                         |
| -------------------------- | --------------------------------- | ---------------- | -------------------------------|
| `run_training()`           | `env`<br>`agent`<br>`total_steps` | –                | 主訓練迴圈                      |
| `sample_level()`           | –                                 | `level: int`     | 依 curriculum / 設定決定訓練關卡 |
| `is_new_game_event()`      | `info dict`                       | `bool`           | 判斷是否發生 win/lose/draw      |
| `find_latest_checkpoint()` | `models_dir`                      | `prefix or None` | 自動接續訓練                    |

### eval_curve_sf.py  
| API | Input | Output | Method |
| ----| ------| -------| -------|
| `list_checkpoints()`  | `models_dir: str` | `List[Tuple[int, str]]`<br>（`[(step, prefix), ...]`）  | 掃描 `models_dir` 內的 `sac_chess_step*_actor.pth` 並解析 step，回傳排序後的 checkpoint 清單 |
| `piece_value()` | `piece_type: int`（chess piece enum）| `int`  | 回傳子力分數（P=1, N/B=3, R=5, Q=9, K=0）|
| `material_diff_white()`  | `board: chess.Board`  | `float` | 計算盤面子力差：白方子力總和 − 黑方子力總和  |
| `plies()` | `board: chess.Board` | `int` | 估算目前半步數（ply count） |
| `filter_legal_actions()` | `legal_actions: List[int]`<br>`action_dim: int` | `List[int]`  | 過濾越界 action（不在 `[0, action_dim)`）避免評估時崩潰 |
| `eval_one_checkpoint()`| `env: ChessBossEnvSF`<br>`agent: SacAgent`<br>`games: int`<br>`eval_level: int`<br>`max_white_moves_per_game: int`<br>`action_dim: int`|`Dict[str, float]`<br>包含：`win_rate/lose_rate/draw_rate`<br>`avg_terminal_material_diff`<br>`avg_terminal_plies`<br>`illegal_count` | 載入指定 checkpoint 權重後，在固定 Level 跑多盤評估，統計勝率、子力差、對局長度與 illegal 次數 |
| `main()` | CLI args：<br>`--models_dir` `--engine_path` `--games` `--level` `--max_white_moves_per_game` `--save_prefix` `--action_dim` | 產出 PNG 檔 + matplotlib 顯示| 批次迭代所有 checkpoint，對固定 Level 做評估並輸出曲線圖（W/L/D、Material、Plies、Illegals）|

### plot_loss_curve.py
| API | Input  | Output | Method  |
| ----| ------ | ------ | ------- |
| `load_loss_csv()` | `path: str`（loss log CSV 路徑）     | `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`<br>依序為：`steps, actor_losses, critic1_losses, critic2_losses, alpha_losses, alpha_values` | 讀取 CSV：支援兩種格式：<br>新版 6 欄：`step,actor_loss,critic1_loss,critic2_loss,alpha_loss,alpha_value`<br>舊版 5 欄：缺 `alpha_value` 則填 `NaN`；忽略空行/格式錯誤行                                                                |
| `main()` | 無（固定呼叫 `load_loss_csv()` 使用預設路徑） | 產出 PNG 檔 + matplotlib 視窗顯示  | 畫並存圖：<br>`critic_loss_curve_sf_v3.png`（critic1/critic2）<br>`actor_loss_curve_sf_v3.png`（actor）<br>`alpha_loss_curve_sf_v3.png`（alpha loss）<br>若 `alpha_values` 不是全 NaN，額外存：`alpha_value_curve_sf_v3.png` |

### demo_gui_sf.py
| API                       | Input                           | Output       | Method                         |
| ------------------------- | ------------------------------- | ------------ | -------------------------------|
| `ChessDemoGUI.__init__`   | `models_dir`<br>`step_delay_ms` | GUI instance | 初始化 Demo 與模型              |
| `start()`                 | –                               | –            | 開始 / 繼續展示                 |
| `pause()`                 | –                               | –            | 暫停                           |
| `step_once()`             | –                               | –            | 單步執行                       |
| `_do_one_env_step()`      | –                               | –            | 執行一次 env → agent → render  |
| `render_board_from_env()` | –                               | –            | 依 env.board 畫棋盤            |
| `run()`                   | –                               | –            | 啟動 GUI loop                  |


Loss Function  
---
### Critic Loss (Q的loss)  
放在sac_agent.py的_update_once()中
```
q1 = critic1(states).gather(actions)
q2 = critic2(states).gather(actions)
```
```math
Q_1(s_t,a_t),Q_2(s_t,a_t)
```
```
target_q = r + (1-done) * gamma * V(next_state)
```
```math
y_t = r_t + (1-done_t) \gamma V(s_{t+1})
```
其中done表示終止狀態，用來告訴Critic這步有沒有未來價值可以期待。  
若非終止狀態(done = 0) : $y_t = r_t + \gamma V(s_{t+1})$  
為終止狀態(done = 1) : $y_t = r_t$

```
critic1_loss = mse(q1, target_q)
critic2_loss = mse(q2, target_q)
```

```math
L_{Q_i} = 𝔼_{(s,a,r,s')}[(Q_i(s,a) - y_t)^2], i ϵ {1,2}
```

### Actor Loss (Policy的objective，用來最小化)  
放在sac_agent.py的_update_once()中  
```
actor_loss = (probs * (self.alpha * log_probs - q_pi)).sum(dim=1).mean()
```
這是 SAC 的 policy objective（希望高 Q、同時保留Entropy)  
因為 optimizer 是做 minimize，所以寫成 alpha*logπ - Q 的形式（越小越好）  
### Critic target  
放在sac_agent.py的_update_once()中   

```
v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
target_q = rewards + (1.0 - dones) * self.gamma * v_next
```
  
```math
V(s') = \sum _a \pi(a|s')(min(Q_1(s',a),Q_2(s',a)) - \alpha log \pi (a|s'))
```
模型成果
---
<img width="640" height="490" alt="螢幕擷取畫面 2026-01-06 220740" src="https://github.com/user-attachments/assets/36ebf6c6-4bef-4720-a860-0be549dbac90" />
<img width="618" height="550" alt="螢幕擷取畫面 2026-01-07 025227" src="https://github.com/user-attachments/assets/940aeff8-3e9d-43a9-9a5b-36efdb9f1da7" />
<img width="640" height="480" alt="actor_loss_curve_sf_v3" src="https://github.com/user-attachments/assets/165fe6e6-fae2-4e9f-8e88-dde6150358d3" />
<img width="640" height="480" alt="critic_loss_curve_sf_v3" src="https://github.com/user-attachments/assets/cefb07d4-4969-4f24-a2a7-32249be2edef" />

demo影片 : [![SAC Chess Boss Demo](https://img.youtube.com/vi/7P4ckps1lrI/0.jpg)](https://youtu.be/7P4ckps1lrI)




