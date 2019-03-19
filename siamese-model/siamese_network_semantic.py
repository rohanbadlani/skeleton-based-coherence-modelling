import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    
    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units, batch_size):
        n_hidden=hidden_units
        n_layers=2
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # fw_cell = tf.contrib.rnn.AttentionCellWrapper(
                #     fw_cell, attn_length=40, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            # lstm_fw_cell_m = tf.contrib.rnn.AttentionCellWrapper(
            #         lstm_fw_cell_m, attn_length=40, state_is_tuple=True)    
            # init_state = lstm_fw_cell_m.zero_state(batch_size, tf.float32)
            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)

            # logits = tf.layers.dense(outputs, n_hidden, name='attn')
            # # logits = tf.matmul(attention_layer, outputs)
            # weights = tf.nn.softmax(logits, name="attention_weights")
            # attention_output = tf.matmul(weights, outputs)
            
        return outputs

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, trainableEmbeddings):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
        print (self.embedded_words1.shape)
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1_list = self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units, batch_size)
            self.out1_perm = tf.convert_to_tensor(self.out1_list)
            
            self.out1_init = tf.transpose(self.out1_perm, perm=[1, 0, 2])
            
            self.out2_list = self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units, batch_size)
            self.out2_perm = tf.convert_to_tensor(self.out2_list)
            self.out2_init = tf.transpose(self.out2_perm, perm=[1, 0, 2])

            
            self.out1_reshape = tf.reshape(self.out1_init, [-1, sequence_length * hidden_units])
            self.out2_reshape = tf.reshape(self.out2_init, [-1, sequence_length * hidden_units])

            self.logits_1 = tf.nn.relu(tf.contrib.layers.fully_connected(
                self.out1_reshape, sequence_length, activation_fn=None))
            # logits = tf.matmul(attention_layer, outputs)
            self.attn_weights_1 = tf.nn.softmax(self.logits_1)
            # self.out1_reshape = tf.reshape(self.out1, [batch_size, sequence_length, hidden_units])
            self.attn_weights_1 = tf.reshape(self.attn_weights_1, [-1, 1, sequence_length])
            self.attention_output_1 = tf.matmul(self.attn_weights_1, self.out1_init)
            self.attn_out1 = tf.reshape(self.attention_output_1, [-1, hidden_units])

            self.logits_2 = tf.contrib.layers.fully_connected(
                self.out2_reshape, sequence_length, activation_fn=None)
            # logits = tf.matmul(attention_layer, outputs)
            self.attn_weights_2 = tf.nn.softmax(self.logits_2)
            # self.out1_reshape = tf.reshape(self.out1, [batch_size, sequence_length, hidden_units])
            self.attn_weights_2 = tf.reshape(self.attn_weights_2, [-1, 1, sequence_length])
            self.attention_output_2 = tf.matmul(self.attn_weights_2, self.out2_init)
            self.attn_out2 = tf.reshape(self.attention_output_2, [-1, hidden_units])

            self.out1 = self.attn_out1
            self.out2 = self.attn_out2
            
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
