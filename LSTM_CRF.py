import tensorflow as tf
import numpy as np
import sklearn.metrics as sk

class LSTM_CRFModel():
    def __init__(self,rnn_size,embedding_size,learning_rate,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vocab,keep_prob):
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.vocab_size = len(sentence_vocab_to_int)
        self.tags_size = len(tags_vocab_to_int)
        self.keep_prob = keep_prob
        self.idx_to_tar = tags_int_to_vocab
        self.buildmodel()

    def buildmodel(self):
        self.inputs  = tf.placeholder(tf.int32,[None,None],name="inputs")
        self.targets = tf.placeholder(tf.int32,[None,None],name="targets")
        self.inputs_length = tf.placeholder(tf.int32,[None],name="inputs_length")
        self.targets_length = tf.placeholder(tf.int32,[None],name="targets_length")
        self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")

        with tf.variable_scope("bilstm"):
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
            embedding_matrix =  tf.get_variable("embedding_matrix",dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[self.vocab_size,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding_matrix,self.inputs)
            outputs,state = tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs,self.inputs_length,dtype=tf.float32)
            outputs = tf.concat(outputs,2)

        with tf.variable_scope("dense"):
            outputs = tf.reshape(outputs,[-1,2*self.rnn_size])
            w_dense = tf.get_variable("w_dense",dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),
                                      shape = [2 *self.rnn_size,self.tags_size])
            b_dense = tf.get_variable("b_dense",dtype=tf.float32,
                                      initializer=tf.zeros_initializer(),
                                      shape=[self.tags_size])
            outputs = tf.matmul(outputs,w_dense)+b_dense
            outputs = tf.reshape(outputs,[self.batch_size,-1,self.tags_size])


        with tf.variable_scope("CRF"):
            log_likelihood,trans = tf.contrib.crf.crf_log_likelihood(
                outputs,self.targets,self.inputs_length
            )
            self.loss = tf.reduce_mean(-log_likelihood)


        with tf.variable_scope("acc"):
            mask = tf.sequence_mask(self.inputs_length)
            viterbi_seq, viterbi_score = tf.contrib.crf.crf_decode(outputs, trans, self.inputs_length)
            output = tf.boolean_mask(viterbi_seq, mask)
            label = tf.boolean_mask(self.targets, mask)
            correct_predictions = tf.equal(tf.cast(output, tf.int32), label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
            self.pred = tf.reshape(viterbi_seq, [-1, ])

        with tf.variable_scope("summary"):
            tf.summary.scalar("trainloss", self.loss)
            tf.summary.scalar("acc", self.accuracy)
            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=None))

        with tf.variable_scope("optimize"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict = {
            self.inputs : batch.inputs,
            self.inputs_length:batch.inputs_length,
            self.targets:batch.targets,
            self.targets_length:batch.targets_length,
            self.batch_size:len(batch.inputs)
        }
        _,loss,summary,acc = sess.run([self.train_op,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)
        return loss,acc,summary

    def vali(self,sess,batch):
        feed_dict = {
            self.inputs: batch.inputs,
            self.inputs_length: batch.inputs_length,
            self.targets: batch.targets,
            self.targets_length: batch.targets_length,
            self.batch_size: len(batch.inputs)
        }

        pred, loss, summary, acc = sess.run([self.pred, self.loss, self.summary_op, self.accuracy],
                                               feed_dict=feed_dict)
        labels = []
        preds = []
        for i, label in enumerate(batch.targets):
            labels.append(label[:batch.targets_length[i]])
        # print(labels)

        pred = np.reshape(pred, [len(batch.inputs), -1, ])
        for i, p in enumerate(pred):
            preds.append(p[:batch.targets_length[i]])
        # print(np.shape(preds[1]))

        labels = [self.idx_to_tar[ii] for lab in labels for ii in lab]
        preds = [self.idx_to_tar[j] for lab_pred in preds for j in lab_pred]
        # weight_f1 = sk.f1_score(labels, preds, average='weighted')
        # micro_f1 = sk.f1_score(labels, preds, average='micro')
        _, _, micro_f1, _ = sk.precision_recall_fscore_support(labels, preds, average='micro')
        _, _, f1, _ = sk.precision_recall_fscore_support(labels, preds, average='weighted')

        return acc, loss, summary, f1,micro_f1