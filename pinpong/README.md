# Pingpong
## 方法 
  - PPO(Proximal Policy Optmization) 
## Loss Function
  - **Policy Loss $`=-L^{CLIP}(\theta)`$**
      - **PPO 理論上是在最大化$`L^{CLIP}`$（讓好動作機率變大、壞動作變小）。**
      - **但深度學習框架（PyTorch）通常用 optimizer 做的是 最小化 loss。**
      - **所以把要「最大化的目標」加一個負號，變成「最小化」。**
    ---
     ### 先定義機率比值(policy ratio)  
     ### $`r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}=\exp\left(\log\pi_\theta(a_t|s_t)-\log\pi_{\theta_{old}}(a_t|s_t)\right) `$
    ---
    ### $`t`$
     - **時間步長（time step），代表 rollout 中的第$`t`$個 frame。**  
    ### $`s_t`$（state / observation）
     - **在第 $`t`$ 步觀測到的狀態**
    ### $`a_t`$（action）
     - **agent 在狀態 $`s_t `$下選到的動作（離散動作，例如 LEFT/RIGHT/NONE/SERVE…）。**
    ### $`\pi_\theta(a_t|s_t)`$（current policy probability）
     - **目前（更新中的）policy 在狀態$`s_t`$下，選到動作$`a_t`$的機率**  
     - **$`\theta`$是「目前模型參數」**
    ### $`\pi_{\theta_{\text{old}}}(a_t|s_t)`$（old policy probability）
     - **收集資料當下舊policy 在狀態$`s_t`$下，選到動作$`a_t`$的機率**
     - **$`\theta_{\text{old}}`$是「收資料時的舊參數」**
    ### $`r_t(\theta)`$（policy ratio）
     - **新舊 policy 對同一個動作的機率比值：**
       - **$`r_t>1`$:新 policy 比舊 policy 更偏好 這個動作**
       - **$`r_t<1`$:新 policy 比舊 policy 更不偏好 這個動作**
       - **PPO 用它來控制更新幅度，避免策略一次改太多。**
    ### $`\log \pi_\theta(a_t|s_t)`$、$`\log \pi_{\theta_{\text{old}}}(a_t|s_t)`$（log probability）
     - **直接算機率容易 underflow（很小的數相乘），所以 PPO 都用 log 空間：**
       - **比值用減法： $`\log \pi_\theta - \log \pi_{\theta_{old}}`$**
       - **再用exp 轉回比值**
    ### $`\exp(\cdot)`$
      - **指數函數，把「log 空間的差」轉回「機率比值」：**
      - **$`\exp(\log p - \log q) = \frac{p}{q}`$**
      ---
    ### Clipped surrogate objective  
    ### $`L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]`$
    ---
    ### $`\mathbb{E}_t[\cdot]`$
      - **代表對$`t`$做平均（期望）**
      - **實作就是：你在 rollout 裡收了很多步資料，把每一步算出來的值 取平均（mini-batch mean）。**
    ### $`\min(\cdot,\cdot)`$
      - **這個 min 是 PPO 的核心安全機制之一。**
      - **它會在「未剪裁版本」與「剪裁版本」之間選比較保守（較小）的那個，避免更新變得太激進。**
      - **如果更新方向對你有利（會讓 objective 變大）但幅度過頭，min 會把你壓回安全範圍。**
    ### $`A_t`$（Advantage）
      - **這一步動作「比平均好多少」的分數。**
        - $`A_t>0$：這動作比基準好 → 希望提高它的機率
        - $`A_t>0$：這動作比基準差 → 希望降低它的機率
    ### $`\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)`$
      - **clip 是把$`r_t`$限制在一個安全區間：**
          - $`r_t \in [1-\epsilon,\; 1+\epsilon]`$
      - **代表「新策略」相對「舊策略」對這個動作的機率最多只能放大到$`1+\epsilon`$、最多只能縮小到$`1-\epsilon`$**
    ### $`\epsilon`$（clip range）
      - **$`\epsilon`$控制「允許策略改變的幅度」。**
      - **越大：更新越激進、可能學更快但更不穩**
      - **越小：更新更保守、穩但可能慢**
    ---
  - **Value Loss $`=\mathbb{E} \left[ (V_{\theta}(s_t) - \hat{R}_t)^2 \right]`$**  
      - **目標是準確預測狀態的長期回報（Returns）。ˋ這裡使用了 均方誤差 (Mean Squared Error, MSE) 來計算預測值與實際回報之間的差距**
    ### $`V_{\theta}(s_t)`$：模型在 $`t`$ 對狀態 $`s_t`$ 預測的狀態價值
    ### $`\hat{R}_t`$：目標回報（Target Return），由 GAE（廣義優勢估計）計算得出的回報值
      - **$`\hat{R}_t = A_t^{GAE} + V_{\theta}(s_t)`$**
    ### $`A_t^{GAE} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}`$
      - $A_t^{GAE}$ ：廣義優勢環境給予的即時獎勵估計值。代表在狀態 $s_t$ 下採取特定動作比平均狀況「好多少」。
      - $r_t$環境給予的即時獎勵
      - $V(s_t)$神經網路預測的當前狀態價值。
      - 時間差分殘差 (TD Residual):首先計算每個時間步的 $\delta_t$（TD Error）：$`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$
      - $\gamma$折扣因子,用於控制對未來獎勵的重視程度。
      - $\lambda$ GAE 混合因子
        - 當 $\lambda = 0$ 時，GAE 退化為 1-step TD。
        - 當 $\lambda = 1$ 時，GAE 變成 Monte Carlo 估計（高變異數）。
  - **Total  Loss**
      $`L^{total} = L^{CLIP} + c_1 \cdot L^{VF} - c_2 \cdot S[\pi_{\theta}](s_t)`$
      - **$L^{total}$：總損失（Total Loss）**
      - **$L^{CLIP}$：策略損失 (Policy Loss)**
      - **$c_1$：價值損失係數（Value Function Coefficient）**
      - **$L^{VF}$：價值損失(Value Loss)**
      - **$c_2$：熵係數（Entropy Coefficient**
      - **$`S[\pi_{\theta}](s_t)`$：策略熵 (Policy Entropy)。用於衡量動作分佈的不確定性，增加熵可以鼓勵模型進行更多探索，避免過早陷入局部最佳解。**
    - **1P Loss plot**
    - <img width="320" height="240" alt="1P_policy_loss" src="https://github.com/user-attachments/assets/1c6f7de3-0eb6-47ee-9719-6972fccaf580" /> <img width="320" height="240" alt="1P_value_loss" src="https://github.com/user-attachments/assets/2aa21673-303f-414c-8377-fd307e26e261" /><img width="320" height="240" alt="1P_total_loss" src="https://github.com/user-attachments/assets/31c6f008-3aa4-4e5a-959e-fba75678915d" /><img width="320" height="240" alt="1P_entropy" src="https://github.com/user-attachments/assets/e2fa2b49-f89c-4816-848a-95ff20d583da" />
    - **2P Loss plot**
    - <img width="320" height="240" alt="2P_policy_loss" src="https://github.com/user-attachments/assets/19f87e2c-bb5a-4674-b60f-a4c531d52aa5" /><img width="320" height="240" alt="2P_value_loss" src="https://github.com/user-attachments/assets/456f6fa3-6f98-40ba-a71e-c4f1adf11178" /><img width="320" height="240" alt="2P_total_loss" src="https://github.com/user-attachments/assets/fea85e20-a572-4b5b-97d6-ad93c186614c" /><img width="320" height="240" alt="2P_entropy" src="https://github.com/user-attachments/assets/c9c2c5b9-b3d5-4e3e-9c7f-abf4b96208c0" />








## 系統規格  
  ### 作業系統:Windows11
  ### Python版本:3.9以上
  ### 使用套件
     NumPy（數值運算、狀態表示）
     Pytorch (2.5.1)
     MLGame + pygame（實際對打 / 視覺化
     Matplotlib（訓練結果分析）
## 系統拆解(Breakdown)
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6df812c7-fd8b-4e8f-89be-6e95faa44ae4" />

## API(主要函式使用說明)
  ### API from train.py
  |Function |Input|Output|使用方法|
  | :--------: | :-----: |:------:|:-------|
  |ActorCritic() |ball_x, ball_y,<br>ball_vx,ball_vy,<br>self_paddle_x,opponent_paddle_x,<br>blocker_x,<br>served_flag|logits:float<br>value:float|以狀態向量作為輸入，透過共享網路提取特徵後，同時輸出動作策略的 logits（Policy）以及狀態價值估計（Value）。|
  |compute_gae()|rewards, values, dones,last_value, gamma, lam|advantages:np.ndarray<br>returns:np.ndarray|算每一步的 Advantage 與 Return，以降低 policy gradient 的方差並提升訓練穩定性|
  | ppo_update()| model, optimizer, obs_tensor, act_tensor,ret_tensor, adv_tensor,oldlog_tensor|none|用一批 rollout 資料，對 Actor-Critic 模型做多輪 PPO 更新（policy + value)|
 ### API from ml_play_pytorch_XP.py
  |Function |Input|Output|使用方法|
  | :--------: | :-----: |:------:|:-------|
  |update( )|scene_info: dict|action:str|MLGame會自動每幀呼叫 <br>1.讀 scene_info<br>2.決策<br>3.回傳指令字串|
  |_is_served() |scene_info: dict|True or False:bool|兼容不同版本 scene_info 格式|
  |_make_obs()|scene_info: dict|obs: np.ndarray|內容為 normalize 後的 8 維狀態向量|
  |_select_action()|obs|act:int|推論動作選擇|
  |forward()|x|logits: torch.Tensor|用 logits 來選動作|
## 流程圖說明  
 ### 資料收集&訓練
   <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c121680f-b298-435a-9a38-4e1f0161c947" />  
   
 ### 推論
 <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6b3f7ada-e032-482a-a258-035650e80d4b" />  


## 驗收  
 ### 功能驗收
  - 1P/2P使用.pt模型
  - 基本功能:1P/2P在遊戲啟動後不會報錯跳出，且能進行發球與擊球
 ### 目標功能
  - 1P/2P能進行數回合對打或是在得分數上55開
## demo  


https://github.com/user-attachments/assets/affa4f68-93d6-4386-8024-a6d662658cf8


    
