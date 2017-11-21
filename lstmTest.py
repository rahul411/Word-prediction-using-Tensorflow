import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
from keras.preprocessing import sequence
import numpy as np

nData = pickle.load(open('data','rb'))
trainData = nData.trainGrams
valData = nData.validGrams
model_path = 'model/model-15'

vocab_size = 8000
n_epochs = 1000
learning_rate = 0.0005

# number of units in RNN cell
dim_hidden = 128
batch_size = 16
n_lstm_steps = 4 + 2
maxlen = 4
embedding_size = 16
graph = tf.Graph()
with graph.as_default():

    sentence = tf.placeholder(tf.int32, [1, maxlen])   #n_lstm_steps is 30 +2 
    # mask = tf.placeholder(tf.float32, [batch_size, n_lstm_steps])
    current_emb = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
    # onehot_labels = tf.placeholder(tf.int32,[batch_size, embedding_size])
    embed_word_W = tf.Variable(tf.random_normal([dim_hidden, vocab_size]))
    embed_word_b = tf.Variable(tf.random_normal([vocab_size]))

    with tf.device("/cpu:0"):
            Wemb = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), name='Wemb')

    bemb = tf.Variable(tf.zeros([embedding_size]), name='bemb')

    # sentence = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
    lstm = BasicLSTMCell(dim_hidden,state_is_tuple=False)
    # state = tf.zeros([batch_size, lstm.state_size])
    def model():
        
        state = tf.zeros([1, lstm.state_size])
        # loss = 0.0
        with tf.variable_scope("RNN"):
                # tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                        current_emb = tf.nn.embedding_lookup(Wemb, sentence[:,-1]) + bemb


                output, state = lstm(current_emb, state) # (batch_size, dim_hidden)

                
                logit_words = tf.matmul(output, embed_word_W) + embed_word_b # (batch_size, n_words)
                max_prob_word = tf.argmax(logit_words, 1)

        return max_prob_word

    max_prob_word = model()
    

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess,model_path)
    data = valData[0]
    data = np.reshape(data,(1,4))
    print(data)
    # print()

    max_prob_word = sess.run([max_prob_word], feed_dict={
            sentence:data})
    # print(word)
    # print(len(word))
    # print(logit_words)
    print(max_prob_word)
    

