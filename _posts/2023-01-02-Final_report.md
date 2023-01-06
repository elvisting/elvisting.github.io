---
layout: post
title: 運用DRQN及DDDQN訓練遊戲之強化學習
author: [mochi_pancake_elvisting]
category: [AI]
tags: [jekyll, ai, reinforce_learning]
math: true
img_path: 
---

# 期末專題實作:運用DRQN及DDDQN訓練遊戲之強化學習

---

## 組員
00953150 鄭丞恩  
00953128 丁昱鈞  
00953101 李承恩

---

## **研究動機及目的**
### *研究動機：*
起初，我們想實作老師提供的題目"四足機器狗之強化學習"，在安裝軟體(REX_GYM)的過程中遇到了些問題，因為我們使用Windows系統，所以必須額外裝微軟的開發套件，導致我們只有部分電腦安裝成功，除此之外，在實際摸索過後，我們只透過內建指令，跑出幾個訓練集，更換場景與姿勢等，但還是不太了解這個軟體如何訓練機器狗，是否需要實體機器，以及如何達成當初的構想，走出迷宮或是跑到指定定點等，為了更了解強化學習的過程，我們決定從頭研究，並藉由"用python實作強化學習：使用 TensorFlow 與 OpenAl Gym"這本書的內容幫助我們學習。

### *研究目的：*
藉由實作DRQN學習Doom，註解範例程式碼，並比較他人製作的DDDQN學習法，使自己深入了解強化學習的運作以及程式設計。
## **理論與演算法**
---
### 策略函數
根據目前狀態，決定執行的動作，說明在各狀態中應該要執行的動作，通常表示為 $ \pi(s):S->A $ 以下介紹三種策略：
1. Stochastic Polic: $a \sim \pi (a\mid s)=P(a \mid s)，s \in S$
2. Deterministic Policy: $a = \pi(s)$(ex: greedy )
3. Random Policy: $a = rand(A)$ ，行為的選擇是隨機的(ex: $\epsilon-greedy$ )

而最佳策略函數我們通常用 $\pi^* \rightarrow argmaxE(r \mid \pi)$ 表示，透過長期觀察回饋值，並進行平均計算得到。
### 獎勵函數與折扣因子
根據於行的動作給予獎勵 $R_t$，譬如在射擊遊戲中，成功擊殺目標給+1，損失血量或子彈則-1，決定動作的好壞，而代理會試著去最大化從環境中得到的獎勵總數(累積獎勵)，通常用 $R_t$ 表示，完整定義如下：
$$R_t = r_{t+1}+r_{t+2}+r_{t+3}+ \cdots +r_T \text{，其中T表示最後一次轉移得到的回饋}$$
但當T趨近於無限大，則加總也會趨近於無限大，這並不是我們想要的，因此我們加入折扣因子(discount factor)的概念，決定當前獎勵與未來將勵的重要程度，數值介於0到1之間，0表示當前獎勵比較重要，1則代表未來獎勵比較重要，通常我們會讓數值介於0.2到0.8之間避免上述情況發生，完整數學式為：
$$R_t = r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+ \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}，\text{where} 0 \leq \gamma \leq 1$$
### 狀態價值函數
判別運用策略 $\pi$ 後在某個狀態中的良好程度。通常用 $V(s)$ 來表示，意思是遵循某個策略後的狀態值。
定義為：
$$V^\pi(s)=\Bbb{E}_\pi \left[R_t \mid s_t=s \right]$$
帶入$R_t$則變成：
$$V^\pi(s)=\Bbb{E}_\pi \left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t=s \right]$$
### Q-Function：動作(action)價值函數

指明代理運用策略 $\pi$ 後，在狀態中執行某個動作的良好程度，Q函數定義為：

$$Q^\pi(s,a)=\Bbb{E}_\pi \left[R_t \mid s_t=s, a_t = a \right]$$
帶入$R_t$則變成：

$$Q^\pi(s,a)=\Bbb{E}_\pi \left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t=s \right]$$

### Bellman Function--Dynamic programming & Monte Carlo & Temporal-Difference
定義完價值函數，我們已經可以找到狀態與動作的價值，現在要來做最佳化，也就是找到最佳策略，我們可以將狀態價值函數與動作價值函數稍加整理與推導，詳細過程可以參考[The Bellman Equation-V-function and Q-function Explained](https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7)，下方列出狀態價值函數與動作價值函數推導後結果：
$$
V^\pi(s)=\sum_{a}\pi(s,a)\sum_{s'}\cal{P}^a_{ss'}\left[R^a_{ss'}+\gamma \it{V}^\pi(s')\right]
$$
$$
Q^\pi(s,a)=\sum_{s'}\cal{P}^a_{ss'}\left[R^a_{ss'}+\gamma\sum_{a'} \it{Q}^\pi(s',a')\right]
$$
其中$\cal{P}^a_{ss'}$為執行動作$a$從狀態$s$移動到$s'$的轉移機率：
$$\cal{P}^a_{ss'}=pr(s_{t+1}=s' \mid s_t{} = s, a_t=a)$$
$\cal{R}^a_{ss'}$為執行動作$a$從狀態$s$移動到$s'$，收到的獎勵機率：
$$\cal{R}^a_{ss'}=\Bbb{E}(\it{R_{t+\rm{1}}}) \mid s_t = s,s_{t+1}=s', a_t=a)$$
由於直接求法過於複雜，因此可以使用下列三種方法來求最佳策略。
* 動態規劃(Dynamic programming，DP)
* 蒙地卡羅(Monte Carlo，MC)
* 時序差分(Temporal-Difference，TD)

動態規劃是一種用於處理複雜問題的技巧。將問題拆成比較簡單的子問題，並計算每個子問題的解決方案。如果發生同樣的子問題，將不會重新計算，直接採納既有方案，降低運算時間。<br>
我們可以用價值迭代或是策略迭代來解Bellman Function，以下是步驟流程圖(擷取自書中)<br>
---
![策略迭代](/graph/policy_itter.jpg){: w="400" h="700" .left}

![價值迭代](/graph/policy_itter.jpg){: w="400" h="700" .right}
---
動態學習必須在轉移機率與獎勵機率已知得前提下運作，因此當我們無法得知環境的模型時，就可以使用MC演算法;當不具備環境知識時，它非常適合用來搜尋最佳策略。
MC透過隨機取樣來找到約略的方案，每個狀態對應各自的獎勵(i.e$V^\pi(s_a)\leftrightarrow R_a$)，並透過執行多條路徑的方法來估計某格結果的機率，基於大數法則的實證方法，當實驗的次數越多，它的平均值也就會越趨近於理論值。我們只需要各狀態、動作與獎勵的取樣順序即可進行，它只適用於世代型(episode)的任務，不需要任何模型，因此也稱為無模型學習法。
缺點就是要花費大量時間(玩到遊戲結束)，因為每次的結果都不一樣(變異數很大)，此外如果將品質不佳的參數和限制輸入到模型中，則會影響輸出結果。

最後要介紹時序差分(TD)它結合了兩者的優點。如同MC，TD學習無須用到模型動態就能運作 ; 它也像DP一樣，不必等到每次世代結束才能估計價值函數。反之，它會根據上次所學的估計值來推測當下的估計值，又稱為自助抽樣法(bootstrapping)。
在MC中，我們運用平均值來估計價值函數，而在TD學習，我們使用當前狀態來更新先前狀態的值，數學式如下：
$$
V^\pi(s_{t-1})=V(s_{t-1})+\alpha(r+\gamma V(s_t)-V(s_{t-1}))
$$
*前一個狀態的價值 = 前一個狀態的價值 + 學習率 ( 獎勵 + 折扣因子
( 當下狀態價值 ) - 前一個狀態的價值 )*
換句話說就是實際獎勵($r+\gamma V(s_t)$)與期望獎勵($V(s_{t-1})$)之差乘以學習率$\alpha$
### Epsilon-greedy Algorithm
為了解決Q-Learning在某一個狀態(state)選擇行為(action)時，會依據前次經驗(Exploitation)找到的最佳解，只進行特定行為，而不會去嘗試其他行為，而錯失其他更好的行為，比如說我們使用的DOOM遊戲，要是一開始機器往左走時可以躲避攻擊並擊殺目標，往後機器也只會往左走，這對我們來說並不樂見，因為或許在某些時候其他的行為會是更好的，為了有更好的探索(Exploration)模式，我們引入ε-貪婪策略(Epsilon-greedy Algorithm)，使機器ε的機率下隨機選擇，在1-ε的機率下由Q-Learning決定行為，通常ε的值不會太大，且會隨時間遞減，使機器在找到最佳行為的情況下，減少隨機選擇的機會。<br>
$$Action\ at\ time\ t\ a(t)= \begin{cases} argmaxQ(s, a), & \text {with probability } 1-\epsilon \\\text{random,} & \text{otherwise} \end{cases}$$


而詳細證明可以參考網站：<https://zhuanlan.zhihu.com/p/63643022> <br>
或是：<https://stats.stackexchange.com/questions/248131/epsilon-greedy-policy-improvement> <br>
可以看出使用此策略可以在Q-learning上有更好的表現<br>
### Q-learning

### Deep-Q-Learning

### DRQN

### DoubleDQN & DuelingDQN


## **系統介紹與程式碼**

由agent透過深度學習遊玩Doom，學習方法有DRQN和DDDQN，獎勵計算方式為成功擊殺怪物得到正項獎勵如果失血或損失子彈則是得到負向獎勵，期望最後的total_reward越來越高。


---
### *系統方塊圖*


### *演算法模型說明*
#### DRQN演算法與程式碼：
引入所需的函式庫：
```python
import tensorflow as tf
import numpy as np
import math
import vizdoom as viz
from tensorboardX import SummaryWriter
```
定義 get_input_shape 函式來計算輸入影像經過卷積層處理後的最後外形：
```python
def get_input_shape(Image,Filter,Stride):
    layer1 = math.ceil(((Image - Filter + 1) / Stride))
    o1 = math.ceil((layer1 / Stride))
    layer2 = math.ceil(((o1 - Filter + 1) / Stride))
    o2 = math.ceil((layer2 / Stride))
    layer3 = math.ceil(((o2 - Filter + 1) / Stride))
    o3 = math.ceil((layer3  / Stride))
    return int(o3)
```
DRQN演算法：
```python
class DRQN():
    def __init__(self, input_shape, num_actions, inital_learning_rate):
        # 初始化所有超參數
        self.tfcast_type = tf.float32
        
        # 設定輸入外形為(length, width, channels)
        self.input_shape = input_shape  
        
        # 環境中的動作數量
        self.num_actions = num_actions
        
        # 神經網路的學習率
        self.learning_rate = inital_learning_rate
                
        # 定義卷積神經網路的超參數

        # 過濾器大小
        self.filter_size = 5
        
        # 過濾器數量
        self.num_filters = [16, 32, 64]
        
        # 間隔大小
        self.stride = 2
        
        # 池大小
        self.poolsize = 2        
        
        # 設定卷積層形狀
        self.convolution_shape = 
        get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]
        
        # 定義循環神經網路與最終前饋層的超參數
        
        # 神經元數量
        self.cell_size = 100
        
        # 隱藏層數量
        self.hidden_layer = 50
        
        # drop out 機率
        self.dropout_probability = [0.3, 0.2]

        # 最佳化的超參數
        self.loss_decay_rate = 0.96
        self.loss_decay_steps = 180

        # 初始化CNN的所有變數

        # 初始化輸入的佔位，形狀為(length, width, channel)
        self.input = tf.compat.v1.placeholder(shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = self.tfcast_type)
        
        # 初始化目標向量的形狀，正好等於動作向量
        self.target_vector = tf.compat.v1.placeholder(shape = (self.num_actions, 1), dtype = self.tfcast_type)

        # 初始化三個回應過濾器的特徵圖
        self.features1 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, input_shape[2], self.num_filters[0]),
                                     dtype = self.tfcast_type)
        
        self.features2 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[0], self.num_filters[1]),
                                     dtype = self.tfcast_type)
                                     
        self.features3 = tf.Variable(initial_value = np.random.rand(self.filter_size, self.filter_size, self.num_filters[1], self.num_filters[2]),
                                     dtype = self.tfcast_type)
        # 初始化RNN變數
        # 討論RNN的運作方式
        self.h = tf.Variable(initial_value = np.zeros((1, self.cell_size)), dtype = self.tfcast_type)
        
        # 隱藏層對隱藏層的權重矩陣
        self.rW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            high = np.sqrt(6. / (self.convolution_shape + self.cell_size)),
                                            size = (self.convolution_shape, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # 輸入層對隱藏層的權重矩陣
        self.rU = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        
        # 隱藏層對輸出層的權重矩陣
        self.rV = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (2 * self.cell_size)),
                                            high = np.sqrt(6. / (2 * self.cell_size)),
                                            size = (self.cell_size, self.cell_size)),
                              dtype = self.tfcast_type)
        # 偏差
        self.rb = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)
        self.rc = tf.Variable(initial_value = np.zeros(self.cell_size), dtype = self.tfcast_type)

        # 定義前饋網路的權重與偏差
        
        # 權重
        self.fW = tf.Variable(initial_value = np.random.uniform(
                                            low = -np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            high = np.sqrt(6. / (self.cell_size + self.num_actions)),
                                            size = (self.cell_size, self.num_actions)),
                              dtype = self.tfcast_type)              
        # 偏差
        self.fb = tf.Variable(initial_value = np.zeros(self.num_actions), dtype = self.tfcast_type)

        # 學習率
        self.step_count = tf.Variable(initial_value = 0, dtype = self.tfcast_type)
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate,       
                                                   self.step_count,
                                                   self.loss_decay_steps,
                                                   self.loss_decay_steps,
                                                   staircase = False)
        # 建置網路

        # 第一卷積層
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), 
        filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool2d(self.relu1, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 第二卷積層
        self.conv2 = tf.nn.conv2d(input = self.pool1, filter = self.features2, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu2 = tf.nn.relu(self.conv2)
        self.pool2 = tf.nn.max_pool2d(self.relu2, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 第三卷積層
        self.conv3 = tf.nn.conv2d(input = self.pool2, filter = self.features3, strides = [1, self.stride, self.stride, 1], padding = "VALID")
        self.relu3 = tf.nn.relu(self.conv3)
        self.pool3 = tf.nn.max_pool2d(self.relu3, ksize = [1, self.poolsize, self.poolsize, 1], strides = [1, self.stride, self.stride, 1], padding = "SAME")

        # 加入 dropout 並重新設定輸入外形
        self.drop1 = tf.nn.dropout(self.pool3, self.dropout_probability[0])
        self.reshaped_input = tf.reshape(self.drop1, shape = [1, -1])

        # 建置循環神經網路，會以卷積網路的最後一層作為輸入
        self.h = tf.tanh(tf.matmul(self.reshaped_input, self.rW) + tf.matmul(self.h, self.rU) + self.rb)
        self.o = tf.nn.softmax(tf.matmul(self.h, self.rV) + self.rc)

        # 在RNN中加入dropout
        self.drop2 = tf.nn.dropout(self.o, self.dropout_probability[1])
        
        # 將RNN的結果送給前饋層
        self.output = tf.reshape(tf.matmul(self.drop2, self.fW) + self.fb, shape = [-1, 1])
        self.prediction = tf.argmax(self.output)

        # 計算損失
        self.loss = tf.reduce_mean(tf.square(self.target_vector - self.output))
        
        # 使用 Adam 最佳器將誤差降到最低
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        
        # 計算損失的梯度並更新梯度
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.update = self.optimizer.apply_gradients(self.gradients)

        self.parameters = (self.features1, self.features2, self.features3,
                           self.rW, self.rU, self.rV, self.rb, self.rc,
                           self.fW, self.fb)
```
定義ExperienceReplay類別來實作經驗回放緩衝，取樣經驗來訓練網路：
```python
class ExperienceReplay():
    def __init__(self, buffer_size):
        
        # 儲存轉移的緩衝
        self.buffer = []       
        
        # 緩衝大小
        self.buffer_size = buffer_size
        
    # 如果緩衝滿了就移除舊的轉移
    # 可把緩衝視佇列，新的進來時，舊的就出去
    def appendToBuffer(self, memory_tuplet):
        if len(self.buffer) > self.buffer_size: 
            for i in range(len(self.buffer) - self.buffer_size):
                self.buffer.remove(self.buffer[0])     
        self.buffer.append(memory_tuplet)  
        
    # 定義 sample 函式來隨機取樣n個轉移  
    def sample(self, n):
        memories = []
        for i in range(n):
            memory_index = np.random.randint(0, len(self.buffer))       
            memories.append(self.buffer[memory_index])
        return memories
```
定義用於訓練網路的train函式：
```python
def train(num_episodes, episode_length, learning_rate, scenario = "deathmatch.cfg", map_path = 'map02', render = False):
  
    # 計算Q值的折扣因子
    discount_factor = .99
    
    # 更新緩衝中經驗的頻率
    update_frequency = 5
    store_frequency = 50
    
    # 顯示輸出結果
    print_frequency = 1000

    # 初始化儲存總獎勵及總損失的變數
    total_reward = 0
    total_loss = 0
    old_q_value = 0

    # 初始化儲存世代獎勵的清單
    rewards = []
    losses = []

    # 動作表現
   
    # 初始化遊戲環境
    game = viz.DoomGame()
    
    # 指定情境檔案路徑
    game.set_doom_scenario_path(scenario)
    
    # 指定地圖檔案路徑
    game.set_doom_map(map_path)

    # 設定螢幕解析度與格式
    game.set_screen_resolution(viz.ScreenResolution.RES_256X160)    
    game.set_screen_format(viz.ScreenFormat.RGB24)

    # 設定以下參數為 true or false 來加入粒子與效果
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)

    # 指定代理可用的按鈕，動作選擇向左、向右以及射擊
    # game.add_available_button(viz.Button.TURN_LEFT)
    # game.add_available_button(viz.Button.TURN_RIGHT)
    # game.add_available_button(viz.Button.MOVE_FORWARD)
    # game.add_available_button(viz.Button.MOVE_BACKWARD)
    game.add_available_button(viz.Button.MOVE_LEFT)
    game.add_available_button(viz.Button.MOVE_RIGHT)
    game.add_available_button(viz.Button.ATTACK)
    
    # 加入一個名為 delta 的按鈕
    # 上述按鈕作用類似鍵盤，所以只會回傳布林值
    # 使用 delta 按鈕來模擬滑鼠來回傳正負數，有助於探索環境
    
    game.add_available_button(viz.Button.TURN_LEFT_RIGHT_DELTA, 90)
    game.add_available_button(viz.Button.LOOK_UP_DOWN_DELTA, 90)

    # 初始化動作陣列
    actions = np.zeros((game.get_available_buttons_size(), game.get_available_buttons_size()))
    count = 0
    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    # 遊戲變數，彈藥、生命值與殺敵數
    game.add_available_game_variable(viz.GameVariable.AMMO0)
    game.add_available_game_variable(viz.GameVariable.HEALTH)
    game.add_available_game_variable(viz.GameVariable.KILLCOUNT)

    # 設定 episode_timeout 在數個時間步驟後停止該世代
    # 另外設定 episode_start_time，有助於跳過初始事件
    game.set_episode_timeout(6 * episode_length)
    game.set_episode_start_time(14)
    game.set_window_visible(render)
    
    # 設定 set_sound_enable 啟動或關閉音效
    game.set_sound_enabled(False)

    # 設定生存獎勵為-1，動作假如無實際作用，則利用扣分迫使代理轉換動作
    game.set_living_reward(-1)

    # doom 有多種模式，像是 玩家(player), 旁觀者(spectator), 非同步玩家(asynchronous player) and 非同步旁觀者(asynchronous spectator)
    # 在旁觀者模式，人類來玩遊戲，代理從中學習
    # 在玩家模式，代理會實際玩遊戲，因此設定為玩家模式
    game.set_mode(viz.Mode.PLAYER)

    # 初始化遊戲環境
    game.init()

    # 建立一個DRQN類別的實例，以及actor與目標DRQN網路
    actionDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    targetDRQN = DRQN((160, 256, 3), game.get_available_buttons_size() - 2, learning_rate)
    
    # 建立一個 ExperienceReplay 類別的實例class，緩衝大小為 1000
    experiences = ExperienceReplay(1000)

    # 儲存模型
    saver = tf.compat.v1.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    # 開始訓練過程
    # 初始化由經驗緩衝中取樣與儲存轉移的變數
    sample = 4
    store = 50

    # 設定 tensorboadX 與 想要觀察的變數 
    writer = SummaryWriter(log_dir = 'runs/' + scenario)
    kill_count = np.zeros(10) # This list will contain kill counts of each 10 episodes in order to compute moving average
    ammo = np.zeros(10) # This list will contain ammo of each 10 episodes in order to compute moving average
    rewards2 = np.zeros(10)
    losses2 = np.zeros(10)
   
    # 開始 tensorflow 階段
    with tf.compat.v1.Session() as sess:
        
        # 初始化所有 tensorflow 變數
        sess.run(tf.global_variables_initializer())
        
        for episode in range(num_episodes):
            
            # 開始新世代
            game.new_episode()
            
            # 在世代中進行遊戲直到世代結束
            for frame in range(episode_length):
                
                # 取得遊戲狀態
                state = game.get_state()
                s = state.screen_buffer
                
                # 選擇動作
                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
                action = actions[a]
                
                # 執行動作與儲存獎勵
                reward = game.make_action(action)
                
                # 更新總獎勵
                total_reward += reward

                # 如過世代結束則中斷迴圈
                if game.is_episode_finished():
                    
                    # tensordroad紀錄輸出
                    kill_count[episode%10] = game.get_game_variable(viz.GameVariable.KILLCOUNT)
                    ammo[episode%10] = game.get_game_variable(viz.GameVariable.AMMO2)
                    rewards2[episode%10] = total_reward
                    losses2[episode%10] = total_loss
                    # 更新 tensordroad writer
                    if (episode > 0) and (episode%10 == 0):
                        writer.add_scalar('Game variables/Kills', kill_count.mean(), episode)
                        writer.add_scalar('Game variables/Ammo', ammo.mean(), episode)
                        writer.add_scalar('Reward Loss/Reward', rewards2.mean(), episode)
                        writer.add_scalar('Reward Loss/loss', losses2.mean(), episode)
                    break
                 
                # 將轉移儲存到經驗緩衝中
                if (frame % store) == 0:
                    experiences.appendToBuffer((s, action, reward))

                # 從經驗緩衝中取樣經驗       
                if (frame % sample) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]
                    
                    # 開始訓練網路
                    Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                    Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input: mem_frame})

                    # 設定學習率
                    learning_rate = actionDRQN.learning_rate.eval()

                    # 計算Q值
                    Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor * Q2 - old_q_value)    
                    
                    # 更新舊的Q值
                    old_q_value = Qtarget

                    # 計算損失
                    loss = actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    
                    # 更新總損失
                    total_loss += loss

                    # 更新兩個網路
                    actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

            rewards.append((episode, total_reward))
            losses.append((episode, total_loss))

            
            print("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))


            total_reward = 0
            total_loss = 0
            
            # tensorbroad 存取輸出並結束
            writer.export_scalars_to_json("./all_scalars.json")
            writer.close()
```
#### DDDQN演算法與程式碼：
原本想使用老師提供的C51演算法，進行結果比較，但嘗試了很久都沒有成功執行，猜測因為版本太舊，vizdoom無法順利安裝，在搜尋網路過後，發現此程式碼，理論上此程式碼遠勝於我們，是一個好的模仿對象。<br>
DDDQN全名為Double Dueling Deep Q-Learning Network，是兩個演算法的結合，參考網址為：<br>
[Deep Reinforcement learning Applied to DOOM](https://github.com/cactuar3101/Deep-Reinforcement-Learning-applied-to-DOOM) \[Fork\]<br>

## **成果、結論與未來展望**
---
### *成果展示*

### *結論與未來展望*
DRQN-->DARQN
運用其他的強化學習方式
## **心得** 
---
### 李承恩

### 丁昱鈞

### 鄭丞恩

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
