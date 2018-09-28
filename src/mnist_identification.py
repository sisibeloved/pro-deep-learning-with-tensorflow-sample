# 导入需要使用的库
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# 导入 MINST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 批量学习参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 50
num_train = mnist.train.num_examples
num_batches = (num_train // batch_size) + 1
epochs = 2

# RNN LSTM 网络参数
n_input = 28  # MNIST 数据输入（图像形状：28*28）
n_steps = 28  # 时间步长
n_hidden = 128  # 隐含特征层的数量
n_classes = 10  # MNIST 总分类（0-9 的数字）


# 定义 RNN 的前向通过
def RNN(x, weights, biases):
    # 出栈以获得 n_steps 个形状为(batch_size, n_input)的张量的列表，如图 4-12 所述
    x = tf.unstack(x, n_steps, 1)

    # 定义 LSTM 细胞
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # 获取 LSTM 细胞输出
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # 线性激活, 使用 RNN 内部循环的最后一个输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


# tf 图像输出
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 定义权重
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = RNN(x, weights, biases)

# 定义损失和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0

    while i < epochs:
        for step in xrange(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # 进行优化操作（反向传播）
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if (step + 1) % display_step == 0:
                # 计算该批次准确率
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # 计算该批次损失
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print "Epoch: " + str(i + 1) + ",step:" + str(step + 1) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc)
        i += 1
    print "Optimization Finished!"

    # 计算准确率
    test_len = 500
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})

