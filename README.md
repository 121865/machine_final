# machine_final
## Ping Pong
- ### 預計使用演算法
#### Proximal Policy Optimization (PPO)
三個Components:Agent、Enviroment、Reward Function  
On-policy:跟環境互動的Agent與訓練的Agent為同一個  
Off-policy:跟環境互動的Agent與訓練的Agent非同一個  

On-policy Gradient:<img width="317" height="39" alt="image" src="https://github.com/user-attachments/assets/2d2482cb-6efc-482e-97fe-809f95270d35" />  

Importance Sampling:<img width="302" height="71" alt="image" src="https://github.com/user-attachments/assets/26dc8fe8-b677-42fe-a3e2-f75fe0b042a6" />

Off-policy Gradient:<img width="421" height="67" alt="image" src="https://github.com/user-attachments/assets/e5344828-b05f-4030-9f90-9f957dd22150" />  

機率拆解如下  
<img width="481" height="68" alt="image" src="https://github.com/user-attachments/assets/92334c15-91b7-47cb-aa92-f9d80c7e393d" />


可以藉由此公式<img width="204" height="36" alt="image" src="https://github.com/user-attachments/assets/aa7e6fc1-3846-4b61-8390-2edcc4c78536" />反推得**Object Function**  
<img width="366" height="62" alt="image" src="https://github.com/user-attachments/assets/5fde6c83-c1e9-4f24-8337-bd493dfe32b5" />  

為了避免theta與theta prime差太多需要加個constraint  
<img width="309" height="56" alt="image" src="https://github.com/user-attachments/assets/72c664d0-3ec4-4c18-b552-16774d41659e" />  
<img width="80" height="31" alt="image" src="https://github.com/user-attachments/assets/c9e464bc-1f3f-4665-a2ec-6826b97e4e56" />是為了判定兩者的behavior或者是action有多像





- ### Loss function
最大Object Function取負=最小化loss  
<img width="559" height="53" alt="image" src="https://github.com/user-attachments/assets/5badf2ac-9f95-4689-907d-e2e7dcbfd950" />  
  
## Chess
- ### 預計使用演算法
#### Soft Actor-Critic (SAC)
簡介 : 前身為Soft Q-learning，因為Soft Q-learning是使用一個函數Q的Boltzmann distribution，在連續空間下求解麻煩  
       所以提出了**Actor**表示策略函數(Policy function)，屬於off-policy。
- ### Loss function
SAC中有兩個動作價值函數Q(參數分別為ω1、ω2)及一個策略函數π(參數為θ)，任意一個**Q的loss function**為:  
<img width="776" height="97" alt="image" src="https://github.com/user-attachments/assets/de78e7c3-f238-4df0-9649-e3fc66b6860e" />  
R : 過去收集的數據  
**策略π的loss function**由KL散度得到 :  
<img width="367" height="39" alt="image" src="https://github.com/user-attachments/assets/916698cc-6b08-4a10-9fed-2aaf27ee72fa" />  
運用**重參數化技巧(reparameterization trick)**及同時考慮兩個函數Q後，重寫策略π的loss function :  
<img width="515" height="67" alt="image" src="https://github.com/user-attachments/assets/d1ccfc25-760c-4ef7-a699-e0f5b92cd4f4" />  
在SAC中，如果在最優狀態不確定的情況下Entropy的取值會盡量取大一點；比較確定的情況下則是取小一點。  
為了能自動調整Entropy正則項，將目標改寫成  
<img width="478" height="62" alt="image" src="https://github.com/user-attachments/assets/002c9efb-45a4-43cd-be81-8893dfc03d49" />  
上述也是最大期望回報，並約束Entropy的值\geq H_0 ，化簡後得到 **α的loss function** :  
<img width="354" height="42" alt="image" src="https://github.com/user-attachments/assets/8997e6ee-4389-4bce-bba9-9d816dd2ab37" />




- ### Reference
<https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr#PPO-algorithm>

