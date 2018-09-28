import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

# ------------------------------------------------------------
# 将单词向量进行独热编码的函数
# ------------------------------------------------------------
def one_hot(ind, vocab_size):
    rec = np.zeros(vocab_size)
    rec[ind] = 1
    return rec

# ----------------------------------------------------------------------
# 从语料库创建训练数据的函数
# ----------------------------------------------------------------------
def create_training_data(corpus_raw, WINDOW_SIZE=2):
    words_list = []

    for sent in corpus_raw.split('.'):
        for w in sent.split():
            if w != '.':
                words_list.append(w.split('.')[0])  # 如果分隔符处于单词的末尾，则将其删除

    words_list = set(words_list)  # 删除重复单词

    word2ind = {}  # 定义把单词转换成索引的字典

    ind2word = {}  # 定义用索引来检索单词的字典

    vocab_size = len(words_list)  # 词汇表中不重复的单词数量

    for i, w in enumerate(words_list):  # 构建字典
        word2ind[w] = i
        ind2word[i] = w

    print(word2ind)
    sentences_list = corpus_raw.split('.')
    sentences = []

    for sent in sentences_list:
        sent_array = sent.split()
        sent_array = [s.split('.')[0] for s in sent_array]
        sentences.append(sent_array)  # 最终 sentences 保存了多个数组，每个数组是由单个句子的单词组成

    data_recs = []  # 用于保存输入输出的记录

    for sent in sentences:
        for ind, w in enumerate(sent):
            for nb_w in sent[max(ind - WINDOW_SIZE, 0): min(ind + WINDOW_SIZE,
                                                            len(sent)) + 1]:
                if nb_w != w:
                    data_recs.append([w, nb_w])

    x_train, y_train = [], []

    for rec in data_recs:
        x_train.append(one_hot(word2ind[rec[0]], vocab_size))
        y_train.append(one_hot(word2ind[rec[1]], vocab_size))

    return x_train, y_train, word2ind, ind2word, vocab_size


corpus_raw = "Deep Learning has evolved from Artificial Neural Networks, which has been there since the 1940s. " \
             "Neural Networks are interconnected networks of processing unitscalled artificial neurons that loosely mimic axons in a biological brain. " \
             "In a biological neuron, the dendrites receive input signals from various neighboring neurons, typically greater than 1000. " \
             "These modified signals are then passed on to the cell body or soma of the neuron, where these signals are summed together and then passed on to the axon of the neuron. " \
             "If the received input signal is more than a specified threshold, the axon will release a signal which again will pass on to neighboring dendrites of other neurons. " \
             "Figure 2-1 depicts the structure of a biological neuron for reference. " \
             "The artificial neuron units are inspired by the biological neurons with some modifications as per convenience. " \
             "Much like the dendrites, the input connections to the neuron carry the attenuated or amplified input signals from other neighboring neurons. " \
             "The signals are passed on to the neuron, where the input signals are summed up and then a decision is taken what to output based on the total input received. " \
             "For instance, for a binary threshold neuron an output value of 1 is provided when the total input exceeds a pre-defined threshold; otherwise, the output stays at 0. " \
             "Several other types of neurons are used in artificial neural networks, and their implementation only differs with respect to the activation function on the total input to produce the neuron output. " \
             "In Figure 2-2 the different biological equivalents are tagged in the artificial neuron for easy analogy and interpretation."

corpus_raw = corpus_raw.lower()
x_train, y_train, word2ind, ind2word, vocab_size = create_training_data(corpus_raw, 2)

# ----------------------------------------------------------------------------
# 定义 TensorFlow 的操作和变量并开始训练
# ----------------------------------------------------------------------------
emb_dims = 128
learning_rate = 0.001
# ---------------------------------------------
# 输入输出的占位符
# ----------------------------------------------
x = tf.placeholder(tf.float32, [None, vocab_size])
y = tf.placeholder(tf.float32, [None, vocab_size])
# ---------------------------------------------
# 定义嵌入矩阵的权重和偏置定义嵌入矩阵的权重和偏置
# ----------------------------------------------
W = tf.Variable(tf.random_normal([vocab_size, emb_dims], mean=0.0, stddev=0.02, dtype=tf.float32))
b = tf.Variable(tf.random_normal([emb_dims], mean=0.0, stddev=0.02, dtype=tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims, vocab_size], mean=0.0, stddev=0.02, dtype=tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size], mean=0.0, stddev=0.02, dtype=tf.float32))

hidden = tf.add(tf.matmul(x, W), b)
logits = tf.add(tf.matmul(hidden, W_outer), b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs, batch_size = 100, 10
batch = len(x_train) // batch_size

# 迭代 n_iter 次
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('was here')
    for epoch in range(epochs):
        batch_index = 0
        for batch_num in range(batch):
            x_batch = x_train[batch_index: batch_index + batch_size]
            y_batch = y_train[batch_index: batch_index + batch_size]
            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
            print('epoch:', epoch, 'loss :', sess.run(cost, feed_dict={x: x_batch, y: y_batch}))
    W_embed_trained = sess.run(W)
W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)
plt.figure(figsize=(10, 10))
for i in range(len(W_embedded)):
    plt.text(W_embedded[i, 0], W_embedded[i, 1], ind2word[i])

plt.xlim(-150, 150)
plt.ylim(-150, 150)
