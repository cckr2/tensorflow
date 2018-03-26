
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import datetime
import os


# In[2]:


class MLP:
    def __init__(self,data_file, log_file, save_file, hidden_node_num, learning_num, save_interval, start_learning_rate = 0.1, decay_rate = 0.96,process_num = -1):
        #Define instance variable
        self.__data_file = data_file
        self.__log_file = log_file
        self.__save_file = save_file
        self.__log = None

        self.__hidden_node_num = hidden_node_num
        self.__hidden_layer_num = len(self.__hidden_node_num)
        
        self.__learning_num = learning_num
        self.__save_interval = save_interval
        self.__start_learning_rate = start_learning_rate
        self.__global_step = tf.Variable(0, trainable=False)
        
        self.__decay_rate = decay_rate
        if(decay_rate == 0):
            self.__learning_rate = self.__start_learning_rate
        else:
            self.__learning_rate = tf.train.exponential_decay(self.__start_learning_rate, self.__global_step, self.__learning_num,
                                                   self.__decay_rate, staircase=True)
        self.__process_num = process_num
            
        self.x_train_data = None
        self.y_train_data = None
        self.x_test_data = None
        self.y_test_data = None
        self.__x_data_len = None
        
        self.__y = None
        self.__x = None
        
        self.__w = []
        self.__b = []
        self.__cost = None
        self.__min_cost = 99999999
        self.__min_acc = 99999999
        
        self.__y_out = None
        self.__train = None
        self.__sess = None
        self.__saver = None
        
        self.__loadData()
        self.__init_model()
        
    def __loadData(self):
        #Read data file
        data_file_name = self.__data_file
        xy = np.genfromtxt(data_file_name, dtype='float32')
        
        #Shuffle data
        np.random.shuffle(xy)
        
        #Data Split into train data and test data
        all_data_num = xy[:,1:-1].shape[0]
        train_data_num = int(all_data_num * 95 /100)
        test_data_num = all_data_num - train_data_num
        self.x_train_data =  xy[:train_data_num,1:-1]
        self.y_train_data =  xy[:train_data_num,-1]
        self.x_test_data =  xy[train_data_num:,1:-1]
        self.y_test_data =  xy[train_data_num:,-1]
        
        #Transpose Matrix having x_data
        self.x_train_data = self.x_train_data.transpose()
        self.x_test_data = self.x_test_data.transpose()

        #Calculate num of variable
        self.__x_data_len = len(self.x_train_data)
        
    def __init_model(self):
        self.__x = tf.placeholder(dtype=tf.float32)
        self.__y = tf.placeholder(dtype=tf.float32)
        
        # first layer
        self.__w.append(tf.Variable(tf.random_normal([self.__hidden_node_num[0], self.__x_data_len]), name="w0"))
        self.__b.append(tf.Variable(tf.random_normal([self.__hidden_node_num[0],1]), name="b0"))
       
        # add hidden layers (variable number)
        for i in range(1,self.__hidden_layer_num):
            wName = "w" + str(i)
            bName = "b" + str(i)
            self.__w.append(tf.Variable(tf.random_normal([self.__hidden_node_num[i], self.__hidden_node_num[i-1]]), name=wName))
            self.__b.append(tf.Variable(tf.random_normal([self.__hidden_node_num[i],1]), name=bName))
        
        # add final layer
        wName = "w" + str(self.__hidden_layer_num)
        bName = "b" + str(self.__hidden_layer_num)
        self.__w.append(tf.Variable(tf.random_normal([1, self.__hidden_node_num[-1]]), name=wName))
        self.__b.append(tf.Variable(tf.random_normal([1],1), name=bName))
        
        # define model
        activate = 1
        drop = True
        drop_rate = 0.85
        if(activate == 0):
            # Sigmoid
            self.__y_out = tf.nn.sigmoid(tf.matmul( self.__w[0],self.__x) + self.__b[0])
            if drop:
                 self.__y_out = tf.nn.dropout(self.__y_out, drop_rate)
            for i in range(1,self.__hidden_layer_num):
                self.__y_out = tf.nn.sigmoid(tf.matmul( self.__w[i],self.__y_out) + self.__b[i])
                if drop:
                    self.__y_out = tf.nn.dropout(self.__y_out, drop_rate)
        if(activate == 1):
            # Relu
            self.__y_out = tf.nn.relu(tf.matmul( self.__w[0],self.__x) + self.__b[0])
            if drop:
                 self.__y_out = tf.nn.dropout(self.__y_out, drop_rate)
            for i in range(1,self.__hidden_layer_num):
                self.__y_out = tf.nn.relu(tf.matmul( self.__w[i],self.__y_out) + self.__b[i])
                if drop:
                    self.__y_out = tf.nn.dropout(self.__y_out, drop_rate)
        
            self.__y_out = tf.matmul(self.__w[-1],self.__y_out) + self.__b[-1]
        
        # setup cost function and optimizer
        cost = 1
        Optimizer = 1
        if(cost == 0):
            # Least Squares
            self.__cost = tf.reduce_mean(tf.square(self.__y_out-  self.__y))
        elif(cost == 1):
            # L2 Loss
            self.__cost = tf.nn.l2_loss(self.__y_out-self.__y)
        elif(cost == 2):
            # Mean squared error
            self.__cost = tf.losses.mean_squared_error(self.__y_out,self.__y)
        elif(cost == 3):
            # Cross entropy
            # Not suitable for airfoil.
            self.__cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__y_out,labels=self.__y))   

        if(Optimizer == 0):
            # Gradient Descent
            opt = tf.train.GradientDescentOptimizer(self.__learning_rate)
        elif(Optimizer == 1):
            # & Adam
            opt= tf.train.AdamOptimizer(self.__learning_rate)
        elif(Optimizer == 2):
            # Adagrad
            opt= tf.train.tf.train.AdagradOptimizer(self.__learning_rate)
        elif(Optimizer == 3):
            # Momentum 
            # MomentumOptimizer with 0.9 momentum is very standard and works well. 
            # The drawback is that you have to find yourself the best learning rate.
            opt= tf.train.tf.train.MomentumOptimizer(self.__learning_rate,momentum=0.9)    
        elif(Optimizer == 4):
            # Adadelta
            # If you really want to use Adadelta, use the parameters (learning_rate=1., rho=0.95, epsilon=1e-6). 
            # A bigger epsilon will help at the start, but be prepared to wait a bit longer than with other optimizers to see convergence.
            opt= tf.train.AdadeltaOptimizer(1., 0.95, 1e-6)
        elif(Optimizer == 5):
            # RMSProp
            # The results are less dependent on a good learning rate. This algorithm is very similar to Adadelta
            opt= tf.train.RMSPropOptimizer(self.__learning_rate)
            
        reg_w = 0
        reg_b = 0
        for i in self.__w:
            reg_w += tf.nn.l2_loss(i)
        for i in self.__b:
            reg_b += tf.nn.l2_loss(i)
            
        cost += reg_w + reg_b
        self.__train = opt.minimize(self.__cost,global_step=self.__global_step)
    
    def init_sess(self):
        init = tf.global_variables_initializer()
        self.__sess = tf.Session()
        self.__sess.run(init)
        
        if self.__save_file != None:
            self.__saver = tf.train.Saver()
            
        if self.__log_file != None:
            self.__log = open(self.__log_file,'w')
            self.__log.write("Step\tTraning Cost\tTest Accuracy\n")
            self.__log.close()
        
    def model_train(self):
        for step in range(self.__learning_num):
            self.__sess.run(self.__train,feed_dict={self.__x: self.x_train_data, self.__y: self.y_train_data})
            if step%self.__save_interval == 0:
                self.__log_write(step)
        self.__log_write(step)
                
    def close(self):
        self.__sess.close();
            
    def __model_save(self,cost,acc):
        if self.__save_file != None:
            if self.__min_cost > cost:
                self.__min_cost = cost
                self.__saver.save(self.__sess, self.__save_file, write_meta_graph=False)
            elif self.__min_cost == cost:
                if self.__min_acc > acc:
                    self.__min_acc = acc
                    self.__saver.save(self.__sess, self.__save_file, write_meta_graph=False)

    def __log_write(self, num):
        if self.__log_file != None:
            predic = self.__sess.run(self.__y_out,feed_dict={self.__x: self.x_test_data})
            acc = np.mean((self.y_test_data- predic) * 100 / predic)
            ccost = self.__sess.run(self.__cost,feed_dict={self.__x: self.x_test_data,self.__y: self.y_test_data})
            self.__log = open(self.__log_file,'a')
            self.__log.write(str(num) + "\t")
            self.__log.write(str(ccost)+ "\t")
            self.__log.write(str(acc)+ "\n")
            self.__log.close()
            self.__model_save(ccost,acc)
    
    def getStep():
        return self.__global_step


# In[7]:


def mytrain(alpha, dataPath, logPath, savePath, hidden_node_num,train_num,save_interval):
    mlp = MLP(dataPath, logPath, savePath, hidden_node_num,train_num,save_interval)
    mlp.init_sess()
    print("Process_",os.getpid(),"(",alpha,") : Train Start")
    mlp.model_train() 
    mlp.close()
    print("Process_",os.getpid(),"(",alpha,") : Train Finish")

