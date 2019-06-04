import tensorflow as tf
from data_util import getBatches,get_data,get_sentence_int_to_vocab,get_posint
from transformer_CRF import LSTM_CRFModel
from tqdm import tqdm

import math
import os
import random

tf.app.flags.DEFINE_integer("rnn_size",100,"Number of hidden units in each layer")
tf.app.flags.DEFINE_integer("num_blocks",1,"Number of transformer block")
tf.app.flags.DEFINE_integer("num_heads",8,"Number of attention head")
tf.app.flags.DEFINE_integer("batch_size",30,"Batch Size")
tf.app.flags.DEFINE_integer("embedding_size",100,"Embedding dimensions of encoder and decoder inputs")
tf.app.flags.DEFINE_float("learning_rate",0.01,"Learning rate")
tf.app.flags.DEFINE_float("keep_prob",0.5,"keep_prob")
tf.app.flags.DEFINE_integer("numEpochs",30,"Maximum # of training epochs")
tf.app.flags.DEFINE_string("model_dir","saves/","Path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","ner.ckpt","File name used for model checkpoints")


FLAGS = tf.app.flags.FLAGS

train_datapath = "dataset/train.txt"
test_datapath = "dataset/test.txt"
sentence_int_to_vocab, sentence_vocab_to_int, tags_vocab_to_int, tags_int_to_vocab = get_sentence_int_to_vocab(train_datapath,test_datapath)
pos_vocab_to_int,pos_int_to_vocab = get_posint()
data = get_data(train_datapath)
train_data = data
vali_data = get_data(test_datapath)


with tf.Session() as sess:
    model = LSTM_CRFModel(FLAGS.num_heads,FLAGS.num_blocks,FLAGS.rnn_size,FLAGS.embedding_size,FLAGS.learning_rate,
                          sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vocab,pos_vocab_to_int,FLAGS.keep_prob)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("reloading model parameters....")
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("create new model parameters...")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    current_step = 0
    avef1list = []
    micf1list = []
    train_summary_writer = tf.summary.FileWriter("saves/train", graph=sess.graph)


    for e in range(FLAGS.numEpochs):

        print("Epoch{}/{}-------------".format(e+1,FLAGS.numEpochs))
        train_batches = getBatches(train_data,FLAGS.batch_size)
        vali_batches = getBatches(vali_data,FLAGS.batch_size)
        for train_Batch in tqdm(train_batches,desc = "Training"):
            trainloss,acc,trainsummary = model.train(sess,train_Batch)
            current_step += 1
            tqdm.write("----Step %d -- trainloss %.2f --acc %.2f " %(current_step,trainloss,acc))
            train_summary_writer.add_summary(trainsummary, current_step)
            if(current_step % 100 == 0):
                ave_f1 = 0
                mic_f1 = 0
                vali_current_step = 0
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)

                model.saver.save(sess, checkpoint_path, global_step=current_step)
                for vali_Batch in tqdm(vali_batches, desc="Vali"):
                    vali_current_step+=1
                    vali_acc,valiloss,valisummary,vali_f1,mirco_f1 = model.vali(sess, vali_Batch)

                    tqdm.write("----Step %d -- valiloss %.2f -- valiacc %.3f --valif1 %.3f --micf1 %.3f " %(vali_current_step,valiloss,vali_acc*100,vali_f1*100,mirco_f1*100))

                    ave_f1+=vali_f1
                    mic_f1+=mirco_f1

                ave_f1 = ave_f1/len(vali_batches)
                mic_f1 = mic_f1/len(vali_batches)
                # if avef1list != [] and ave_f1 * 100 >= max(avef1list):


                print("ave_f1",ave_f1*100)
                print("mic_f1",mic_f1*100)

                avef1list.append(ave_f1*100)
                micf1list.append(mic_f1*100)

print("最大avef1为：",max(avef1list))
print("最大avef1索引为：",avef1list.index(max(avef1list)))
print("最大micf1为：",max(micf1list))
print("最大micf1索引为：",micf1list.index(max(micf1list)))