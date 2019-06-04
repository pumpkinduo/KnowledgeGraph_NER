import tensorflow as tf
import numpy as np
import sklearn.metrics as sk

class LSTM_CRFModel():
    def __init__(self,num_heads,num_blocks,rnn_size,embedding_size,learning_rate,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vocab,pos_vocab_to_int,keep_prob):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.vocab_size = len(sentence_vocab_to_int)
        self.tags_size = len(tags_vocab_to_int)
        self.keep_prob = keep_prob
        self.idx_to_tar = tags_int_to_vocab
        self.pos_vocab_size = len(pos_vocab_to_int)
        self.buildmodel()

    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = 1e-8

        inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值

        numHeads = self.num_heads
        keepProp = 0.9

        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 将每一时序上的向量中的值相加取平均值
        keyMasks = tf.sign(tf.abs(tf.reduce_sum(rawKeys, axis=-1)))  # 维度[batch_size, time_step]

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(keyMasks, [numHeads, 1])

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
                                  scaledSimilary)  # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
                                      maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络

        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.batch_size
        sequenceLen = 30
        embeddingSize = self.embedding_size

        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        #0::2表示从0开始隔两个数取一个值 0 2 4 6 8.....
        #[:,0::2]表示在全部列表里面的偶数列
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded

    def buildmodel(self):
        self.inputs  = tf.placeholder(tf.int32,[None,None],name="inputs")
        self.targets = tf.placeholder(tf.int32,[None,None],name="targets")
        self.inputs_length = tf.placeholder(tf.int32,[None],name="inputs_length")
        self.targets_length = tf.placeholder(tf.int32,[None],name="targets_length")
        self.pos = tf.placeholder(tf.int32,[None,None],name="pos")
        self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")

        with tf.variable_scope("encoder"):
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
            embedding_matrix =  tf.get_variable("embedding_matrix",dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[self.vocab_size,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding_matrix,self.inputs)
            pos_embedding_matrix =  tf.get_variable("pos_embedding_matrix",dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[self.pos_vocab_size,self.embedding_size])
            pos = tf.nn.embedding_lookup(pos_embedding_matrix,self.pos)
            inputs = tf.concat([inputs,pos],axis=-1)
            inputs = tf.reshape(inputs,[self.batch_size,-1,2*self.embedding_size])
            with tf.variable_scope("transformer"):
                for i in range(self.num_blocks):
                    with tf.name_scope("transformer-{}".format(i + 1)):
                        # 维度[batch_size, sequence_length, embedding_size]
                        multiHeadAtt = self._multiheadAttention(rawKeys=inputs, queries=inputs,
                                                                keys=inputs, numUnits=None, causality=False,
                                                                scope="multiheadAttention")
                        # 维度[batch_size, sequence_length, embedding_size]
                        self.embeddedWords = self._feedForward(multiHeadAtt,
                                                               [2*self.rnn_size, 2*self.embedding_size])

                outputs = self.embeddedWords

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
            self.pos: batch.pos,
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
            self.pos :batch.pos,
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