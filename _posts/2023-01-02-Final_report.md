---
layout: post
title: 運用DRQN及DDDQN訓練遊戲之強化學習
authors: [<鄭丞恩>,<李承恩>,<丁昱鈞>]
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
### *研究動機*
起初，我們想實作老師提供的題目"四足機器狗之強化學習"，在安裝軟體(REX_GYM)的過程中遇到了些問題，因為我們使用Windows系統，所以必須額外裝微軟的開發套件，導致我們只有部分電腦安裝成功，除此之外，在實際摸索過後，我們只透過內建指令，跑出幾個訓練集，更換場景與姿勢等，但還是不太了解這個軟體如何訓練機器狗，是否需要實體機器，以及如何達成當初的構想，走出迷宮或是跑到指定定點等，為了更了解強化學習的過程，我們決定從頭研究，並藉由"用python實作強化學習：使用 TensorFlow與OpenAl Gym"這本書的內容幫助我們學習。

### *研究目的*
藉由實作DRQN學習Doom，註解範例程式碼，並比較他人製作的DDDQN學習法，使自己深入了解強化學習的運作以及程式設計。
## **理論與演算法**
此章節陳述我們使用的理論以及演算法。

---
### *策略函數*
根據目前狀態，決定執行的動作，說明在各狀態中應該要執行的動作，通常表示為 $ \pi(s):S->A $ 以下介紹三種策略：
1. Stochastic Polic: $a \sim \pi (a\mid s)=P(a \mid s)，s \in S$
2. Deterministic Policy: $a = \pi(s)$(ex: greedy )
3. Random Policy: $a = rand(A)$ ，行為的選擇是隨機的(ex: $\epsilon-greedy$ )

而最佳策略函數我們通常用 $\pi^* \rightarrow argmaxE(r \mid \pi)$ 表示，透過長期觀察回饋值，並進行平均計算得到。
### *獎勵函數與折扣因子*
根據於行的動作給予獎勵 $R_t$，譬如在射擊遊戲中，成功擊殺目標給+1，損失血量或子彈則-1，決定動作的好壞，而代理會試著去最大化從環境中得到的獎勵總數(累積獎勵)，通常用 $R_t$ 表示，完整定義如下：

$$R_t = r_{t+1}+r_{t+2}+r_{t+3}+ \cdots +r_T \text{，其中T表示最後一次轉移得到的回饋}$$

但當T趨近於無限大，則加總也會趨近於無限大，這並不是我們想要的，因此我們加入折扣因子(discount factor)的概念，決定當前獎勵與未來將勵的重要程度，數值介於0到1之間，0表示當前獎勵比較重要，1則代表未來獎勵比較重要，通常我們會讓數值介於0.2到0.8之間避免上述情況發生，完整數學式為：

$$R_t = r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+ \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}，\text{where} 0 \leq \gamma \leq 1$$

### *狀態價值函數*
判別運用策略 $\pi$ 後在某個狀態中的良好程度。通常用 $V(s)$ 來表示，意思是遵循某個策略後的狀態值。
定義為：

$$V^\pi(s)=\Bbb{E}_\pi \left[R_t \mid s_t=s \right]$$

帶入$R_t$則變成：

$$V^\pi(s)=\Bbb{E}_\pi \left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t=s \right]$$

### *Q-Function：動作(action)價值函數*

指明代理運用策略 $\pi$ 後，在狀態中執行某個動作的良好程度，Q函數定義為：

$$Q^\pi(s,a)=\Bbb{E}_\pi \left[R_t \mid s_t=s, a_t = a \right]$$
帶入$R_t$則變成：

$$Q^\pi(s,a)=\Bbb{E}_\pi \left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \mid s_t=s \right]$$

### *Bellman Function--Dynamic programming & Monte Carlo & Temporal-Difference*
定義完價值函數，我們已經可以找到狀態與動作的價值，現在要來做最佳化，也就是找到最佳策略，我們可以將狀態價值函數與動作價值函數稍加整理與推導，詳細過程可以參考[The Bellman Equation-V-function and Q-function Explained](https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7)，下方列出狀態價值函數與動作價值函數推導後結果：


$$V^\pi(s)=\sum_{a}\pi(s,a)\sum_{s'}\cal{P}^a_{ss'}\left[R^a_{ss'}+\gamma \it{V}^\pi(s')\right]$$

$$Q^\pi(s,a)=\sum_{s'}\cal{P}^a_{ss'}\left[R^a_{ss'}+\gamma\sum_{a'} \it{Q}^\pi(s',a')\right]$$

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

![策略迭代](/graph/policy_itter.jpg){: w="350" h="700" .left}![價值迭代](/graph/policy_itter.jpg){: w="350" h="700" .left}_策略迭代(左)價值迭代(右)_

動態學習必須在轉移機率與獎勵機率已知得前提下運作，因此當我們無法得知環境的模型時，就可以使用MC演算法;當不具備環境知識時，它非常適合用來搜尋最佳策略。
MC透過隨機取樣來找到約略的方案，每個狀態對應各自的獎勵(i.e$V^\pi(s_a)\leftrightarrow R_a$)，並透過執行多條路徑的方法來估計某格結果的機率，基於大數法則的實證方法，當實驗的次數越多，它的平均值也就會越趨近於理論值。我們只需要各狀態、動作與獎勵的取樣順序即可進行，它只適用於世代型(episode)的任務，不需要任何模型，因此也稱為無模型學習法。
缺點就是要花費大量時間(玩到遊戲結束)，因為每次的結果都不一樣(變異數很大)，此外如果將品質不佳的參數和限制輸入到模型中，則會影響輸出結果。

最後要介紹時序差分(TD)它結合了兩者的優點。如同MC，TD學習無須用到模型動態就能運作 ; 它也像DP一樣，不必等到每次世代結束才能估計價值函數。反之，它會根據上次所學的估計值來推測當下的估計值，又稱為自助抽樣法(bootstrapping)。
在MC中，我們運用平均值來估計價值函數，而在TD學習，我們使用當前狀態來更新先前狀態的值，數學式如下：

$$
V^\pi(s_{t-1})=V(s_{t-1})+\alpha(r_{t+1}+\gamma V(s_t)-V(s_{t-1}))
$$

*前一個狀態的價值 = 前一個狀態的價值 + 學習率 ( 獎勵 + 折扣因子
( 當下狀態價值 ) - 前一個狀態的價值 )*

換句話說就是實際獎勵($r_{t+1}+\gamma V(s_t)$)與期望獎勵($V(s_{t-1})$)之差乘以學習率$\alpha$

### *Epsilon-greedy Algorithm*
為了解決Q-Learning在某一個狀態(state)選擇行為(action)時，會依據前次經驗(Exploitation)找到的最佳解，只進行特定行為，而不會去嘗試其他行為，而錯失其他更好的行為，比如說我們使用的DOOM遊戲，要是一開始機器往左走時可以躲避攻擊並擊殺目標，往後機器也只會往左走，這對我們來說並不樂見，因為或許在某些時候其他的行為會是更好的，為了有更好的探索(Exploration)模式，我們引入$\epsilon$-貪婪策略(Epsilon-greedy Algorithm)，使機器$\epsilon$的機率下隨機選擇，在1-$\epsilon$的機率下由Q-Learning決定行為，通常$\epsilon$的值不會太大，且會隨時間遞減，使機器在找到最佳行為的情況下，減少隨機選擇的機會。<br>

$$Action\ at\ time\ t\ a(t)= \begin{cases} argmaxQ(s, a), & \text {with probability } 1-\epsilon \\\text{random,} & \text{otherwise} \end{cases}$$

而詳細證明可以參考網站：<https://zhuanlan.zhihu.com/p/63643022> <br>
或是：<https://stats.stackexchange.com/questions/248131/epsilon-greedy-policy-improvement> <br>
可以看出使用此策略可以在Q-learning上有更好的表現<br>
### *Q-learning*
在課堂上已經介紹過Q-Learning，這裡將在稍加介紹，TD-Learning可以分為兩大類，On-policy與Off-poicy，前者會從資料當中學習到同一種策略($\pi$)，再進行改進(ex:[Sarsa](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action))，Off-poicy沒有固定策略，會利用學習到的經驗根據當前狀態推斷出動作的價值，也就是其學習到的策略是獨立於訓練資料(ex:[Q-Learnig](https://en.wikipedia.org/wiki/Q-learning))，這裡我們只介紹Q-Learning，在Q學習，我們關注的是Q-Function，也就是在狀態s中執行a所產生的效果，我們會根據以下方程式更新Q-value：

$$
Q^{new}(s_t,a_t)\leftarrow Q(s_t, a_t)+\alpha[r_{t+1}+\gamma \underset{a}{\operatorname{\max}}Q(s_{t+1},a)-Q(s_t,a_t)]
$$

*新的Q = 舊的Q + 學習率 ( 獎勵 + 折扣因子
( 最優未來價值的估計 ) - 舊的Q  )*

其詳細步驟為：

![Q-Learning](/graph/q_learning.jpg){: w="350" h="700" .center}_Q-Learning 流程圖_

### *Deep-Q-Network*
簡單的架構圖如下：

![DQN_frame](/graph/DQN_frame.png){: w="350" h="700" .center}_DQN_frame_


### *DQN-algorithm*
Paper：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

Frame&Words discription：

![DQN_algorithm_word](/graph/DQN_algorithm_word.png){: w="350" h="700" .left}![DQN_algorithm](/graph/DQN_algorithm.png){: w="350" h="700" .left}_DQN_algorithm_words(left)&DQN_algorithm(right)_

左邊紅色線為初始狀態的第一步，將 St, at, rt, S(t+1)給算出來並存放至記憶體裡面，第一步初始化做完之後，再進行藍色線的flow，通過環境來儲存St, at, rt, S(t+1)，並將參數丟給對應的網路來計算LOSS Function，最後再更新網路的參數，一直不斷的重覆更新就可以找出最好的Q Function。

#### Environment

會將environment環境每一個時間點的observation(觀察)的集合當作環境的State(狀態)
從環境的狀態(State)跟reward(獎勵)再去選擇一個最好的action(動作)，稱為policy(策略)

#### Replay Memory

深度學習的訓練資料最好為獨立同分布的資料，
然而RL的資料是時序性的，也就是代表資料前後是有關聯的，這樣可能會造成模型無法正常訓練，所以建立了一個空間來儲存資料，
並且利用Random 採樣的方式來進行training，這樣就不會有關聯性的問題了。

#### Q網路

經過一些時間後，把訓練用的參數給復製到計算Target Q的網路，用此手法把計算Target Q的神經網路跟訓練用的神經網路分開，


### *DRQN*

DRQN的架構與DQN相當類似，但第一個後卷積完全連接層會換成LSTM RNN，接著把遊戲畫面用於卷積層的輸入。

卷積層會把影像卷積起來好產生特徵圖。這個特徵圖則繼續被送往LSTM層。LSTM層具有存放資訊用的記憶體。

LSTM層會保留關於前一個遊戲狀態的重要資訊，並根據我們的需求來定期更新其記憶。

它會在通過一個完全連接層之後輸出一個Q值。因此與DQN不同，在此不直接去估計$Q(s_t,a_t)$，而是去估計$Q(h_t,a_t)$，而ht是網路在上一個時間步驟所回傳的輸入，也就是說$h_t=LSTM(h_t-1,o_t)$。
由於我們使用的是RNN，我們會透過反向傳播來訓練網路。

在DQN中為了避免經驗彼此關聯會使用經驗回放緩衝中儲存遊戲轉移，
並運用隨機的小批經驗來訓練網路。以DRQN來說，我們在經驗緩衝中儲存了整個世代，並從隨機小批世代中隨機取樣n個步驟。這樣一來就能兼顧隨機性，以及實際彼此跟隨的經驗。

詳細架構如下：

![DRQN_frame](/graph/DRQN_frame.png){: w="350" h="700" .center}_DRQN_frame_

### *DoubleDQN*

DoubleDQN的原始演算法為：[Double Q-learning” (Hasselt, 2010)](https://arxiv.org/pdf/1509.06461.pdf)

![DoubleQ-learning](/graph/DRQN_frame.png){: w="350" h="700" .center}_DoubleQ-learning_

進階版的DoubleDQN：[Deep Reinforcement Learning with Double Q-learning” (Hasselt et al., 2015)](https://arxiv.org/pdf/1509.06461.pdf)

顧名思義就是運用兩個Q函數、 各自獨立學習 。 一個函數是用來選擇動作 · 而另一個 Q 函數則是用來評估動作，可以解決估計Q值時因為雜訊，導致某個動作的評價變高，影響結果。詳細的演算法如下圖：

![DoubleDQN](/graph/DoubleDQN.jpg){: w="350" h="700" .center}_DoubleDQN_

### DuelingDQN

定義為Q函數價值函數兩者之差，代表相較於其他動作，代理執行動作$a$的良好程度，其架構如下：

![DuelingDQN_frame](/graph/DuelingDQN_frame.jpg){: w="350" h="700" .center}_DuelingDQN_frame_

(以下內容擷取自書中)

競爭 DQN 的架構基本上與 DQN 是差不多的，主要差別在於未端的全連接層分成了兩道流 (stream)。一道流是計算價值函數。另一個則負責計算優勢函數。最後，我們運用聚合 (aggregate) 層來結合這兩個流並取得 Q 函數 。
為什要把 Q 函數的運算拆成兩道流呢？在許多狀態中 · 算出所有動作的估計值不是很重要的事情 ，尤其是當狀態的動作空間很大時；代表多數動作對於該狀態根本沒有影像。 再者 ， 也會有許多動作的影響是重複的 。 以這些狀況來說 ， 競爭 DQN 就能比現有的 DQN 架構更精準地來估計 Q 值 ：

* 第一道流 ， 又稱價值流，適用於狀態中的動作數量非常多，以及估計各動作值並不是非常重要的情況 。
* 第二道流 ， 又稱優勢流 ， 適用於網路需要決定偏好哪個動作的情況。

聚合層會整合這兩道流的數值並產生 Q 函數，這就是為什麼競爭網路會比標準 DQN 架構來得更有效率也更強健的原因 。

## **系統介紹與程式碼**

由agent透過深度學習遊玩Doom，學習方法有DRQN和DDDQN，獎勵計算方式為成功擊殺怪物得到正項獎勵如果失血或損失子彈則是得到負向獎勵，期望最後的total_reward越來越高。

---
### *系統方塊圖*
**TRAIN**

![Train](/graph/train_flow.png){: w="350" h="700" .center}

**DRQN**

![DRQN](/graph/DRQN_flow.png){: w="350" h="700" .center}

### *演算法模型說明*
#### [DRQN演算法與程式碼](https://github.com/Williamochi/AI-project/blob/gh-pages/DRQN_programming)
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
def train(num_episodes, episode_length, learning_rate, scenario = "basic.wad", map_path = 'map01', render = False):
  
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
    
    # 建立一個 ExperienceReplay 類別，緩衝大小為 1000
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

                # 如果世代結束則中斷迴圈
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
            # 存取總獎勵及總損失
            rewards.append((episode, total_reward))
            losses.append((episode, total_loss))

            print("Episode %d - Reward = %.3f, Loss = %.3f." % (episode, total_reward, total_loss))

            total_reward = 0
            total_loss = 0
            
            # tensorbroad 存取輸出並結束
            writer.export_scalars_to_json("./all_scalars.json")
            writer.close()
```
執行 train 程式來進行訓練
```python
train(num_episodes = 500, episode_length = 300, learning_rate = 0.0001, render = True)
```
#### DDDQN演算法與程式碼：
原本想使用老師提供的C51演算法，進行結果比較，但嘗試了很久都沒有成功執行，猜測因為版本太舊，vizdoom無法順利安裝，在搜尋網路過後，發現此程式碼，理論上此程式碼遠勝於我們，是一個好的模仿對象。<br>
DDDQN全名為Double Dueling Deep Q-Learning Network，是兩個演算法的結合，參考網址為：<br>
[Deep Reinforcement learning Applied to DOOM](https://github.com/cactuar3101/Deep-Reinforcement-Learning-applied-to-DOOM) \[Fork\]<br>

## **成果、結論與未來展望**

此章節展示我們的實作結果，對結果的論述以及未來目標。

---

### *成果展示*

#### DRQN V.S. DDDQN(趨勢圖)

![DRQN_linechart_1](/graph/DRQN_linechart_1.png){: w="350" h="700" .center}

![DRQN_linechart_2](/graph/DRQN_linechart_2.png){: w="350" h="700" .center}_DRQN-linechart_

![DDDQN_linechart_1](/graph/DDDQN_linechart_1.png){: w="350" h="700" .center}

![DDDQN_linechart_2](/graph/DDDQN_linechart_2.png){: w="350" h="700" .center}_DDDQN-linechart_

根據DDDQN的趨勢圖所顯示，彈藥使用量以及獎勵皆有上升趨勢，以及損失有下降趨勢，代表此演算法有助於訓練此遊戲。

DRQN的趨勢圖在彈藥的使用量及獎勵皆有上升趨勢，但損失趨勢因為程式編寫關係導致呈現上升，總體來說，DRQN演算法也有助於訓練此遊戲。

在趨勢圖上能看出DDDQN的上升穩定度及效能比DRQN來得穩定且較佳，這符合當初的預期。

#### DRQN程式運行圖片

![DRQNgame_termianl](/graph/DRQNgame_termianl.png){: w="350" h="700" .center}_DRQN_

#### DDDQN程式運行圖片

![DDDQNgame_termianl](/graph/DDDQNgame_termianl.png){: w="350" h="700" .center}_DDDQN_

#### 程式運行影片
[運用DRQN演算法遊玩遊戲實作影片](https://youtu.be/28RMXN0h_5o)

{% include embed/youtube.html id='28RMXN0h_5o' %}

[運用DDDQN演算法遊玩遊戲實作影片](https://youtu.be/B6kBbGwOOX8)

{% include embed/youtube.html id='B6kBbGwOOX8' %}

### *結論與未來展望*

這次的實作成功達成起初目的，當然要改進的地方還有很多，像是tensorboard的使用，這是在實作程式碼時發現的功能，可以直接將資訊呈現在圖表中，省下畫圖的時間，也讓我們很清楚的觀察實作結果，此外我們確實沒有完整了解演算法及程式的運做，導致在Debug的時候花了不少時間。

未來目標希望可以運用其他強化學習的方法，來訓練Agent去玩DOOM，除了可以認識更多強化學習，也可以練習自己的程式能力，此外也可以改進我們的DRQN演算法，在書中提供了進階的版本--DARQN，並沒有提供任何程式碼，期許我們在了解此演算法後，能嘗試獨力完成，讓自己更熟悉強化學習的世界。

## **心得** 
---
### 李承恩



### 丁昱鈞

這次的期末作業讓我學到很多，首先在強化學習方面有很多的認識，雖然說只是冰山一角，但卻引起我更大的興趣，也發現到，如果我想要在這方面有突破，我的機率學真的是待加強，有了數學式才能有更的演算法，這將會列在我寒假的目標，其次是github pages的使用，我在換主題方面吃了不少苦頭，回頭看才發現，當初只要依照開發者提供的步驟，一切就會是如此的順其自然，而且我使用的這個主題，開發者真的寫得很詳細，之後有機會的話一定會跟朋友分享我的學習心得，而在Markdown語法上，練習到之前學Latex未使用的數學指令，學習到如何插入圖片、影片與超連結等，在查資料的過程也認識了一些HTML語法，唯一美中不足的是，我尚未了解到如何大量更換字型(我只會使用HTML更換部分字體的語法)，期許我自己在完成未來展望的同時，也去學習如何更換字型(我猜測在主題的設定裡面)，由衷感謝有此次期末專題，讓我學到很多。

### 鄭丞恩

在此報告，我們使用DQN演算法以及DRQN演算法來對射擊遊戲進行強化學習，並且透過結果比較這兩者演算法的差異。我負責的部分為執行利用DRQN演算法學習遊戲，並且整理提供DQN及DRQN兩個演算法的資訊並從中學習，從整理資訊以及透過註解程式來理解其中的原理，學習到了強化學習的運作方式以及運算的原理，還有程式邏輯跟程式流程，透過畫面的資訊作為輸入用程式設定獎勵、懲罰和目標，經過學習的結果並計算損失來讓學習達到更好的成果。藉由此次期末報告讓我獲益良多，也從中發掘了我這一方面的興趣，希望未來有機會也能夠繼續深耕並應用所學。

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
