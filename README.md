# machine_final
## Ping Pong
- ### 預計使用演算法
#### Proximal Policy Optimization (PPO)
---
強化學習三個Components:Agent、Enviroment、Reward Function  
On-policy:跟環境互動的Agent與訓練的Agent為同一個  
Off-policy:跟環境互動的Agent與訓練的Agent非同一個  

__On-policy Gradient:__
```math
E_{(s_t, a_t) \sim \pi_\theta} \left[ A^\theta(s_t, a_t) \nabla \log p_\theta(a^n_t \vert s^n_t) \right]
```
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



Off-policy Gradient:

機率拆解如下  
<img width="481" height="68" alt="image" src="https://github.com/user-attachments/assets/92334c15-91b7-47cb-aa92-f9d80c7e393d" />


可以藉由此公式<img width="204" height="36" alt="image" src="https://github.com/user-attachments/assets/aa7e6fc1-3846-4b61-8390-2edcc4c78536" />反推得**Object Function**  
<img width="366" height="62" alt="image" src="https://github.com/user-attachments/assets/5fde6c83-c1e9-4f24-8337-bd493dfe32b5" />  

為了避免theta與theta prime差太多需要加個constraint  
<img width="309" height="56" alt="image" src="https://github.com/user-attachments/assets/72c664d0-3ec4-4c18-b552-16774d41659e" />  
<img width="80" height="31" alt="image" src="https://github.com/user-attachments/assets/c9e464bc-1f3f-4665-a2ec-6826b97e4e56" />是為了判定兩者的behavior或者是action有多像;beta設定  
<img width="528" height="123" alt="image" src="https://github.com/user-attachments/assets/3004c96c-e181-4c94-877d-9c0e2191152d" />  


J




- ### Loss function
最大Object Function取負=最小化loss  
<img width="559" height="53" alt="image" src="https://github.com/user-attachments/assets/5badf2ac-9f95-4689-907d-e2e7dcbfd950" />  
  
## Chess
- ### 預計使用演算法
#### Soft Actor-Critic (SAC)
簡介 : 前身為Soft Q-learning，因為Soft Q-learning是使用一個函數Q的Boltzmann distribution，在連續空間下求解麻煩  
       所以提出了 **Actor** 表示策略函數(Policy function)，屬於off-policy。
- ### 狀態價值函數
<img width="533" height="39" alt="image" src="https://github.com/user-attachments/assets/de5c24f6-20e8-4445-ae9e-8469bf29ead9" />  

- ### Loss function
SAC中有兩個動作價值函數Q(參數分別為ω1、ω2)及一個策略函數π(參數為θ)，任意一個**Q的loss function**為:  
<img width="776" height="97" alt="image" src="https://github.com/user-attachments/assets/de78e7c3-f238-4df0-9649-e3fc66b6860e" />  
R : 過去收集的數據  
**策略π的loss function**由KL散度得到 :  
<img width="367" height="39" alt="image" src="https://github.com/user-attachments/assets/916698cc-6b08-4a10-9fed-2aaf27ee72fa" />  
運用 **重參數化技巧(reparameterization trick)** 及同時考慮兩個函數Q後，重寫策略π的loss function :  
<img width="515" height="67" alt="image" src="https://github.com/user-attachments/assets/d1ccfc25-760c-4ef7-a699-e0f5b92cd4f4" />  
在SAC中，如果在最優狀態不確定的情況下Entropy的取值會盡量取大一點；比較確定的情況下則是取小一點。  
為了能自動調整Entropy正則項，將目標改寫成  
<img width="478" height="62" alt="image" src="https://github.com/user-attachments/assets/002c9efb-45a4-43cd-be81-8893dfc03d49" />  
上述也是最大期望回報，並約束Entropy的值>= H0 ，化簡後得到 **α的loss function** :  
<img width="354" height="42" alt="image" src="https://github.com/user-attachments/assets/8997e6ee-4389-4bce-bba9-9d816dd2ab37" />

- ### Object function
<img width="347" height="65" alt="image" src="https://github.com/user-attachments/assets/cd0d1d68-816f-44e9-957c-11bc5682cc79" />  

- ### 應用  
狀態價值函數 : 衡量當前局面的好壞  
策略函數 : 決定模型在棋盤上的走法選擇傾向  
loss function : LQ​(ω) --> 評價每步棋的好壞  
                Lπ​(θ) --> 輸出每一步棋的機率分佈，若多步棋 Q 值接近entropy 會鼓勵模型繼續探索其他可行走法  
                L(α)  --> 模型太保守、老是走同一套開局則提高 α，強迫嘗試新策略 ；模型太亂、像亂下棋則降低 α，使決策更穩定  
- ### Reference
<https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr#PPO-algorithm>  
<https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95/>

