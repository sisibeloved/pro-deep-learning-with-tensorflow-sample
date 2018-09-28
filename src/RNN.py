# 导入需要使用的库
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

# 参数定义
learning_rate = 0.001
training_iters = 50000
display_step = 500
n_input = 3

# RNN 细胞中的单元数量
n_hidden = 512


# 读取和处理输入文件的函数
def read_data(fname):
    with open(fname) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    data = [data[i].lower().split() for i in range(len(data))]
    data = np.array(data)
    data = np.reshape(data, [-1, ])
    return data


# 构造和翻转单词字典的函数
def build_dataset(train_data):
    count = collections.Counter(train_data).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


# 对输入向量进行独热编码的函数
def input_one_hot(num):
    x = np.zeros(vocab_size)
    x[num] = 1
    return x.tolist()


# 读取输入文件并构造所需的字典
train_file = 'alice in wonderland.txt'
train_data = read_data(train_file)
dictionary, reverse_dictionary = build_dataset(train_data)
vocab_size = len(dictionary)

# 小批量输入输出的占位符
x = tf.placeholder("float", [None, n_input, vocab_size])
y = tf.placeholder("float", [None, vocab_size])

# RNN 输出节点的权重和偏置
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}


# 前向通过循环神经网络
def RNN(x, weights, biases):
    x = tf.unstack(x, n_input, 1)

    # 定义双层 LSTM
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    # 产生预测
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # 共有 n_input 个输出，不过我们只需要最后一个输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# 损失和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 变量初始化
init = tf.global_variables_initializer()

# 绘制图像
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    while step < training_iters:
        if offset > (len(train_data) - end_offset):
            offset = random.randint(0, n_input + 1)

        symbols_in_keys = [input_one_hot(dictionary[str(train_data[i])]) for i in range(offset, offset + n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vocab_size])
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(train_data[offset + n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

        loss_total += loss
        acc_total += acc

        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100 * acc_total / display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [train_data[i] for i in range(offset, offset + n_input)]
            symbols_out = train_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - Actual word:[%s] vs Predicted word:[%s]" % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (n_input + 1)
    print("TrainingCompleted!")
    # 输入一个 3 个单词的句子，让模型预测接下来的 28 个单词
    sentence = 'i only wish'
    words = sentence.split(' ')
    try:
        symbols_in_keys = [input_one_hot(dictionary[str(train_data[i])]) for i in
                           range(offset, offset + n_input)]
        for i in range(28):
            keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vocab_size])
            onehot_pred = session.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(input_one_hot(onehot_pred_index))
        print("Complete sentence follows!")
        print(sentence)
    except:
        print("Error while processing the sentence to be completed")