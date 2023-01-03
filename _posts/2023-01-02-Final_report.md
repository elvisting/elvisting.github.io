---
layout: post
title: 運用DRQN及C15訓練遊戲之強化學習
author: [mochi_pancake_elvisting]
category: [Lecture]
tags: [jekyll, ai]
---

期末專題實作:運用DRQN及C15訓練遊戲之強化學習

---
## 運用DRQN及C15訓練遊戲之強化學習
### 組員
00953150 鄭丞恩  
00953128 丁昱鈞  
00953101 李承恩
### 系統簡介及功能說明
# **系統簡介**:
初步想法:  
1.做出地圖並訓練狗狗走出ex:單一出口的封閉空間  
2.隨機丟出物品並讓狗狗撿取並回到原位
# **功能說明**:
---
### 系統方塊圖

演算法模型說明  
```
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
        self.convolution_shape = get_input_shape(input_shape[0], self.filter_size, self.stride) * get_input_shape(input_shape[1], self.filter_size, self.stride) * self.num_filters[2]
        
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
        
        # hiddent to output weight matrix
                          
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
        self.conv1 = tf.nn.conv2d(input = tf.reshape(self.input, shape = (1, self.input_shape[0], self.input_shape[1], self.input_shape[2])), filter = self.features1, strides = [1, self.stride, self.stride, 1], padding = "VALID")
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
```
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
---
### 製作步驟
1. 建立資料集dataset
2. 移植程式到kaggle
3. kaggle訓練模型
4. kaggle測試模型
---
### 系統測試及成果展示

---



<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
