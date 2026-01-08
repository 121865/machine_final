# Pingpong $``$
## 方法 
  - PPO(Proximal Policy Optmization) 
## Loss Function
  - **Policy Loss $`= -L^{CLIP}(\theta)`$**
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
  - Value  Loss 
  - Total  Loss 
