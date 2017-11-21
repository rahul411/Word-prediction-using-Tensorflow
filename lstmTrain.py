import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
from keras.preprocessing import sequence
import numpy as np
import os

model_path = 'model/'

nData = pickle.load(open('data','rb'))
trainData = nData.trainGrams
valData = nData.validGrams

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

    sentence = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])   #n_lstm_steps is 30 +2 
    mask = tf.placeholder(tf.float32, [batch_size, n_lstm_steps])
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
        
        state = tf.zeros([batch_size, lstm.state_size])
        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(n_lstm_steps): # maxlen + 1
                if i == 0:
                    current_emb = tf.random_normal([batch_size,embedding_size])
                else:
                    with tf.device("/cpu:0"):
                        current_emb = tf.nn.embedding_lookup(Wemb, sentence[:,i-1]) + bemb

                if i > 0 : tf.get_variable_scope().reuse_variables()

                output, state = lstm(current_emb, state) # (batch_size, dim_hidden)

                if i > 0:
                    labels = tf.expand_dims(sentence[:, i], 1) # (batch_size)
                    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
                    concated = tf.concat([indices, labels],1)
                    onehot_labels = tf.sparse_to_dense(
                            concated, tf.stack([batch_size, vocab_size]), 1.0, 0.0) # (batch_size, n_words)

                    logit_words = tf.matmul(output, embed_word_W) + embed_word_b # (batch_size, n_words)
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                    cross_entropy = cross_entropy * mask[:,i]#tf.expand_dims(mask, 1)

                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(mask[:,1:])
            return loss

    loss = model()
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels))    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver(max_to_keep=50)

    tf.initialize_all_variables().run()

    for epoch in range(n_epochs):
        for start, end in zip( \
            range(0, len(trainData), batch_size),
            range(batch_size, len(trainData), batch_size)):

            # current_feats = feats[image_id[start:end]]
            # current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)

            data = trainData[start:end]
            # current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_data_matrix = sequence.pad_sequences(data, padding='post', maxlen=maxlen+1)
            # print(current_caption_matrix)
            current_data_matrix = np.hstack( [np.full( (len(current_data_matrix),1), 0), current_data_matrix] ).astype(int)

            current_mask_matrix = np.zeros((current_data_matrix.shape[0], current_data_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_data_matrix ))
            #  +2 -> #START# and '.'
            # print(nonzeros)
            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1
            # print(current_mask_matrix)
            _, loss_value = sess.run([optimizer, loss], feed_dict={
                sentence:current_data_matrix,
                mask:current_mask_matrix})

            print "Current Cost: ", loss_value

        print ("Epoch ", epoch, " is done. Saving the model ... ")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


