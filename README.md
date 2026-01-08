# machine_final
## 分工
組長 : F114112116 劉正偉 負責乒乓球  
組員 : F114112114 胡家豪 負責西洋棋  

## Ping Pong
- ### 文獻回顧
#### Proximal Policy Optimization (PPO)
---
強化學習三個Components:Agent、Enviroment、Reward Function  
On-policy:跟環境互動的Agent與訓練的Agent為同一個  
Off-policy:跟環境互動的Agent與訓練的Agent非同一個  

__On-policy Gradient:__
```math
E_{(s_t, a_t) \sim \pi_\theta} \left[ A^\theta(s_t, a_t) \nabla \log p_\theta(a^n_t \vert s^n_t) \right]
```

  * $`(a^n_t \vert s^n_t)`$是整個trajectory內的某一個時間點的成對資料(pair)
    * 如果這個pair會導致整個trajectory的reward變大，那就要增加它出現的機率，反之則減少。
  * $`A^\theta(s_t,a_t)`$:在某一個state-$`s_t`$執行某一個action-$`a_t`$，相較於其它可能的action，現在執行的這一個有多好。

<img width="681" height="233" alt="image" src="https://github.com/user-attachments/assets/8b8b1d4d-56b5-42fd-94ee-c702c2305d2e" />

---
__Importance Sampling__:
```math
  E_{x \sim p}\left[f(x) \right]=E_{x \sim q} \left[f(x) \dfrac{p(x)}{q(x)} \right]
```  
* __推導如下:__  
  * $`E_{x \sim p}\left[f(x) \right]\approx \dfrac{1}{N} \sum^N_{i=1} f(x^i)`$  
    * 沒有辦法對distribution-𝑝計算積分，可以用sample的方式，取平均值來近似期望值  
  * $`=\int f(x) p(x) dx`$
    * 對$`p(x)`$計算積分
  * $`=\int f(x) \dfrac{p(x)}{q(x)} q(x) dx`$
    * 分子分母同乘$`q(x)`$
  * $`=E_{x \sim q} \left[f(x) \dfrac{p(x)}{q(x)} \right]`$
    * 調整為從$`q`$來sample出$`x`$取期望值
    * 需要乘上一個權重$`\dfrac{p(x)}{q(x)}`$來修正$`p`$,$`q`$兩個distribution之間的差異



__Off-policy Gradient:__
```math
E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(s_t, a_t)}{P_{\theta'}(s_t, a_t)} A^{\theta'}(s_t, a_t) \nabla \log p_\theta(a^n_t \vert s^n_t) \right]
```
機率拆解如下  
```math
E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} \dfrac{p_\theta(s_t)}{p_{\theta'}(s_t)} A^{\theta'}(s_t, a_t) \nabla \log p_\theta(a^n_t \vert s^n_t) \right]
```
  * 假設模型在$`\theta`$與$`\theta'`$看到$`s_t`$的機率是差不多的，因此刪除。
  * 另一個想法，$`s_t`$難以估測，因此無視。


可以藉由此公式<img width="204" height="36" alt="image" src="https://github.com/user-attachments/assets/aa7e6fc1-3846-4b61-8390-2edcc4c78536" />
反推得**Objective Function**  
```math
  J^{\theta'}(\theta) = E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} A^{\theta'}(s_t, a_t) \right]
```

為了避免$`\theta`$與$`\theta'`$差太多需要加個constraint  
<img width="309" height="56" alt="image" src="https://github.com/user-attachments/assets/72c664d0-3ec4-4c18-b552-16774d41659e" />  
<img width="80" height="31" alt="image" src="https://github.com/user-attachments/assets/c9e464bc-1f3f-4665-a2ec-6826b97e4e56" />散度是為了判定兩者的behavior或者是action有多像;$`\beta`$設定  
<img width="528" height="123" alt="image" src="https://github.com/user-attachments/assets/3004c96c-e181-4c94-877d-9c0e2191152d" />  





- ### Loss function
```math
L(\theta) = - J^{\theta'}(\theta)
```
```math
L(\theta) = - E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \dfrac{P_\theta(a_t \vert s_t)}{P_{\theta'}(a_t \vert s_t)} A^{\theta'}(s_t, a_t) \right]
```
---







  




## Chess
* ### 預計使用演算法
<mark>Soft Actor-Critic (SAC) <mark>  
  
**簡介 :**
前身為Soft Q-learning，因為Soft Q-learning 是一個使用函數Q的Boltzman distribution，在連續空間下求解麻煩，所以提出了**Actor**表示策略函數(Policy Function)，屬於Off-policy。  
  
* ### SAC的Object Function  
  
```math
J(\pi) = 𝔼 _\pi \left[ \sum \limits _{t=0} ^{\infty} \gamma ^t (r(s_t,a_t) + \alpha H  (\pi(\cdot|s_t)))\right]
```
定義 :  
$`J(\pi)`$ : 整個SAC想最大化的目標函數，代表策略 $`\pi`$ 的好壞。  
$`𝔼_\pi [\cdot]`$ : 期望值，代表「照著策略 $`\pi`$ 與環境互動」所得到的平均結果。  
$`\sum \limits _{t=0} ^\infty`$ : 把整個過程所有時間步的回報累加。  
$`\gamma^t`$ : 折扣因子(discoun factor) ，介於0~1之間，越久遠的回報權重越低。  
$`r(s_t,a_t)`$ : reward function，在狀態 $`s_t`$ 做動作 $`a_t`$ 得到的立即回饋。  
$`\alpha H (\pi(\cdot|s_t))`$ : 探索獎勵(entropy bouns)，由 $`\alpha`$ 跟 $`H (\pi(\cdot|s_t))`$ 組成，「行為越多樣化 $`\to`$ entropy越高 $\to$ 探索越多」。  
$`\alpha`$ : 溫度係數(temperature/entropy weight) ，控制entropy的重要程度。 $`\alpha`$ 越大 $\to$ 越鼓勵探索；越小 $`\to`$ 越鼓勵利用。  
$`H (\pi(\cdot|s_t))`$ : policy在 state $`s_t`$ 的entropy ，計算公式等於 $`-𝔼_{a\sim \pi(\cdot|s_t)}\left[log\pi(a_t|s_t) \right]`$。   
* ### Critic Loss (Q-network的loss)
  
```math
L_Q(\omega) = 𝔼_{(s_t,a_t,r_t,s_{t+1})\sim R}\left[{1\over 2}(Q_\omega (s_t,a_t) - y_t)^2  \right]  
```
定義 :  
$`L_Q(\omega)`$ : Q網路要最小化的損失，而 $`\omega`$ 是Q網路的參數(weights)。  
$`𝔼_{(s_t,a_t,r_t,s_{t+1})\sim R}[\cdot]`$ : 從經驗回放緩衝區(Replay Buffer)中隨機取樣一個transition做期望，也就是Q是用off-policy資料訓練。R $`\to`$ 經驗數據的分布或集合。  
$`{1\over 2}(\cdot)^2`$ : 均方誤差(MSE)，希望Q的輸出越接近目標 $`y_t`$。  
$`Q_\omega (s_t,a_t)`$ : 在狀態 $`s_t`$ 執行動作 $`a_t`$ 的預期總回報(含 entropy)，也就是Q-network輸出這個state-action的價值。  
$`y_t`$ : 實際應該要接近的價值(Target value) ，等於 $`r_t + \gamma(\min \limits_j Q_{\bar{\omega}_j}(s_{t+1},a_{t+1}) - \alpha log \pi (a_{t+1}|s_{t+1}) )`$    
$`r_t`$ : 當下reward 。  
$`\gamma`$ : 折扣因子。  
$`\min\limits_j Q_{\bar{\omega}_j}(s_{t+1},a_{t+1})`$ : 用兩個target Q-net的最小值，避免高估(Double Q的技巧)。  
$`\bar{\omega}`$ : target Q-network的參數(慢慢更新的Q，用來穩定訓練)。  
$`\alpha log \pi (a_{t+1}|s_{t+1})`$ : 下一步entropy bouns。  
  
* ### Policy Loss (actor的Loss)
  
```math
L_\pi (\theta) = 𝔼_{s_t \sim R,a_t \sim \pi_\theta} \left[\alpha log \pi_\theta (a_t|s_t) - Q_\omega (s_t,a_t) \right]
```
    
定義 :  
$`L_\pi (\theta)`$ : actor(policy network)要最小化的損失， $`\theta`$ 為 policy network 的參數。  
$`s_t \sim R`$ : 狀態從replay buffer抽樣(off-policy) 。  
$`a_t \sim \pi_\theta(\cdot|s_t)`$ : 在 state $`s_t`$下，從policy $`\pi`$ 取樣動作。  
$`\alpha log \pi_\theta (a_t|s_t)`$ : 越確定的動作機率越接近1 $`\to`$ log $`\pi`$ 越大(負的)，也可說是這項的效果為 **增加entropy且鼓勵行為更隨機** 。  
$`-Q_\omega (s_t,a_t)`$ : 若Q值越大則這項負數大 $`\to`$ 有助於降低loss，鼓勵選擇Q高的行為。  
  
* ###  $\alpha$  Loss
  
```math
L(\alpha) = 𝔼_{a_t \sim \pi} \left[- \alpha log \pi (a_t|s_t) - \alpha H_0 \right]
```
**目的為自動調整 $`\alpha`$ 使 : $`𝔼[-log\pi] = H_0`$，entropy自動維持在希望的水準。**  
  
定義 :  
$`L(\alpha)`$ : 專門用來更新 $`\alpha`$ 的 loss。  
$`- \alpha log \pi (a_t|s_t)`$ : 當策略過於確定(entropy太低)時， $`log\pi`$會變小，loss偏大會推動 $`\alpha`$ 提高 $\to$ 促使策略更隨機。  
$`-\alpha H_0`$ : $H_0$ 是目標entropy，讓策略的entropy朝固定目標靠近。  
  
* ### Reparameterization Function(SAC core) 
  
```math
a_t = f_\theta(\epsilon_t ;s_t) ， \epsilon_t \sim N(0,I)
```

**讓policy抽樣變成可微 $\to$ 可以用backprop訓練actor。**  
  
定義 :  
$`f_\theta`$ : 一個可微分函數，通常是 $`f_\theta(\epsilon,s) = tanh(\mu_\theta(s_t) + \sigma _\theta(s_t) \cdot \epsilon_t)`$ ，包含高斯分布取樣( $`u_t = \mu_\theta(s_t) + \sigma_\theta(s_t) \cdot \epsilon_t`$ ) 跟 tanh縮放( $`a_t = tanh(u_t)`$ )  
$`\epsilon_t \sim N(0,I)`$ : 從標準常態N(0,1) 取的noise，提供隨機性。  
  
* ### Soft Value Function
  
```math
V(s_t) = 𝔼_{a_t \sim \pi} \left[Q(s_t,a_t) - \alpha log \pi(a_t|s_t) \right]
```
  
**$`V(s_t)`$ = 平均「選到的Q值 + 該動作的探索獎勵」。**  
定義 :  
$`V(s_t)`$ : 在狀態 $`s_t`$ 的預期總價值，但soft value不只是reward，也包含 entropy bouns。  
$`𝔼_{a_t \sim \pi}[\cdot]`$ : 由策略 $`\pi`$ 取樣動作。  
$`Q(s_t,a_t)`$ : 該動作的Q-value(回報總期望)。  
$`-\alpha log \pi(a_t|s_t)`$ : 代表探索bouns，越隨機越有獎勵。  

- ### 應用  
狀態價值函數 : 衡量當前局面的好壞  
策略函數 : 決定模型在棋盤上的走法選擇傾向  
loss function : LQ​(ω) --> 評價每步棋的好壞  
                Lπ​(θ) --> 輸出每一步棋的機率分佈，若多步棋 Q 值接近entropy 會鼓勵模型繼續探索其他可行走法  
                L(α)  --> 模型太保守、老是走同一套開局則提高 α，強迫嘗試新策略 ；模型太亂、像亂下棋則降低 α，使決策更穩定  
- ### Breakdown
![breakdown](https://github.com/user-attachments/assets/7e59e893-60b4-4168-ba47-7e828cf34e60)  
- ### API
- #### train_sac_chess_fullgame.py
<img width="733" height="165" alt="image" src="https://github.com/user-attachments/assets/9ebabe2d-ce20-4c76-a0de-b3333c11a329" />  
<img width="734" height="402" alt="image" src="https://github.com/user-attachments/assets/0c7ebe40-efa6-4748-bbd5-32bb60ba3f72" />    

num_steps : 總互動步數  
updates_per_step : 做幾次參數更新  
log_interval : 印出訓練狀況，計算平均loss  
save_intervl : 隔多少global step存一次
resume_prefix : 手動指定
anto_resume : 自動找訓練最新檔
- #### env_chess_boss.py
<img width="405" height="168" alt="image" src="https://github.com/user-attachments/assets/38d623a1-05e5-42a4-97b0-661df3e63611" />  
<img width="346" height="168" alt="image" src="https://github.com/user-attachments/assets/d7495f7b-6ade-43ad-bf09-1c8aec3bcba1" />  
<img width="474" height="166" alt="image" src="https://github.com/user-attachments/assets/f762c2d0-eeb4-44ee-bc9d-ab9b549803c1" />  
<img width="733" height="254" alt="image" src="https://github.com/user-attachments/assets/68a35f9b-4b70-4ed1-871d-49e4d361c6ea" />  
<img width="472" height="167" alt="image" src="https://github.com/user-attachments/assets/87444f95-eb5d-4e32-a709-86a1868ce437" />  
<img width="474" height="225" alt="image" src="https://github.com/user-attachments/assets/c9ba5ae9-979e-4123-9897-a0b8338aaaac" />  
<img width="548" height="248" alt="image" src="https://github.com/user-attachments/assets/e4fc1fb7-0110-42f6-be1c-c461fe689be4" />  
<img width="547" height="250" alt="image" src="https://github.com/user-attachments/assets/d7eb97b7-4ccd-43f1-9786-50637678238a" />  
<img width="548" height="166" alt="image" src="https://github.com/user-attachments/assets/5be16d8e-4d80-4e21-af57-7a03b5d8eabf" />  
<img width="551" height="168" alt="image" src="https://github.com/user-attachments/assets/2c911992-a02a-440a-86c9-f6871864a02f" />  

  
- #### sac_agent.py
<img width="388" height="223" alt="image" src="https://github.com/user-attachments/assets/dbe3a1f9-c6db-4aef-9310-56c9fc1e2f18" />
<img width="389" height="163" alt="image" src="https://github.com/user-attachments/assets/89a1af84-9ae9-4bb9-a50b-3e28309e0210" />  
<img width="443" height="461" alt="image" src="https://github.com/user-attachments/assets/4e0b2853-5921-4366-ae2f-5fb679419f4d" />  
<img width="497" height="433" alt="image" src="https://github.com/user-attachments/assets/afeda22f-0f92-4d04-8cae-c436a7f43959" />  
<img width="496" height="166" alt="image" src="https://github.com/user-attachments/assets/ec34c4d7-4b99-4e66-a6f9-58d5bc9f18f3" />  
<img width="657" height="251" alt="image" src="https://github.com/user-attachments/assets/6bb07c7d-7ad9-487c-927d-5af665e964b7" />  
<img width="494" height="198" alt="image" src="https://github.com/user-attachments/assets/64b939fd-be81-48c4-8160-9fec174892d4" />    

"actor_loss": float,"critic1_loss": float,"critic2_loss": float,"alpha_loss": 0.0, "alpha_value": float(alpha),  
<img width="535" height="315" alt="image" src="https://github.com/user-attachments/assets/7dcb613e-d691-4689-bb0c-561b10bf359d" />  
<img width="542" height="227" alt="image" src="https://github.com/user-attachments/assets/5a1eb769-a24a-4a41-9bad-ea82896a0e52" />  
<img width="663" height="227" alt="image" src="https://github.com/user-attachments/assets/e4ef6a8b-a144-404f-8d11-ccb2bb98c60f" />  

- ### 結果
### step 0 ~ 1800000
**alpha = 0.05 、 tau = 0.005 、 1r = 1e-4 、**   
**win = 1 、 lose = -1 、 draw = -0.4 、**  
**step_penalty = -0.0001 、 material_coeff = 0.02 、 check_bonus = 0.03 、 check_penalty = -0.03**  
### step 1600000 ~ 2600000
**alpha改成0.03、tau改成0.003、1r改成5e-5**  
### step 2600000 ~ 3200000
**win = 1.2、lose = -1.0 、 draw = -0.6 、**  
**alpha = 0.02 、 actor_1r = 1e-4 、 critic_1r = 5e-5**  
  
<img width="640" height="480" alt="actor_loss_curve5" src="https://github.com/user-attachments/assets/d6a13021-8f5c-4283-a0bd-b2aca9d2ff7e" />  
  
<img width="640" height="480" alt="critic_loss_curve5" src="https://github.com/user-attachments/assets/bb9783a4-53dd-43e8-bf14-b85e59d0b4a2" />  
  
<img width="640" height="480" alt="eval_curve_fullgamev5" src="https://github.com/user-attachments/assets/8ad74c70-98d1-4171-9103-6f51ab4e175c" />  


- ### Reference
<https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr#PPO-algorithm>PPO解釋   
<https://medium.com/@kdk199604/ppo-efficient-stable-and-scalable-policy-optimization-15b5b9c74a88>PPO實際應用架構   
<https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95/>

