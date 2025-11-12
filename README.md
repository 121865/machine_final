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

- ### Reference
<https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyez2AmFr#PPO-algorithm>
## Chess
