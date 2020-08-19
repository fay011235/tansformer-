# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:16:01 2020

@author: fay
"""

import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt
import argparse
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import pickle
warnings.filterwarnings("ignore")

from data_compute import Dataset
from models import nextBatch,fixedPositionEmbedding,Transformer,mean,accuracy,\
                       binary_precision,binary_recall,binary_f_beta,multi_precision,multi_recall,multi_f_beta,\
                           get_binary_metrics,get_multi_metrics

import keras.backend as K


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--action', type=str, required=True, help='choose a action:trian or pred')
parser.add_argument('--pred_dir', type=str, required=False, help='choose a action:trian or pred')
parser.add_argument('--train_continue', default=0, type=int, required=False, help='choose a action:trian or pred')
parser.add_argument('--model_graph_id', type=str, required=False, help='choose a action:trian or pred')
parser.add_argument('--model_ckpt_id', default=None, type=str, required=False, help='model_ckpt_id')
parser.add_argument('--model_save_path', type=str, required=True, help='model_save_path')
args = parser.parse_args()

# 配置参数

class TrainingConfig(object):
    epoches = 500
    no_upnum = 20
    evaluateEvery = 100
    checkpointEvery = 10
    learningRate = 0.1


class ModelConfig(object):
    embeddingSize = 60

    filters = 200  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 5  # Attention 的头数
    numBlocks = 3  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 1# 全连接层的keeppout
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 60# 80  # 取了所有序列长度的中位数
    batchSize = 128

    dataSource = 'data/preProcess/embeing_all_train_4.csv' #'data/preProcess/all_time_interwan.csv' #
    devSource = 'data/preProcess/embeing_all_dev_4.csv'
    predSource = args.pred_dir #"data/preProcess/labeledPred.csv"

    stopWordSource = "data/english"

    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.95  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()
    class_weights = None#np.array([[0.06, 0.94]]) #None

'''
def predstep(batchX, batchY):
    """
    预测函数
    """
    feed_dict = {
      transformer.inputX: batchX,
      transformer.inputY: batchY,
      transformer.dropoutKeepProb: 1.0,
      transformer.embeddedPosition: embeddedPosition
    }
    summary, step, loss, predictions = sess.run(
        [summaryOp, globalStep, transformer.loss, transformer.predictions],feed_dict)

    return predictions
'''

if __name__ == '__main__':
    print('开始')

    # 实例化配置参数对象
    config = Config()

    data = Dataset(config)

    data.dataGen(args.action)

    if args.action == 'train':
        # 训练模型
        # 生成训练集和验证集
        trainReviews = data.trainReviews
        trainLabels = data.trainLabels

        evalReviews = data.evalReviews
        evalLabels = data.evalLabels
        lens = config.batchSize

    elif args.action == 'pred':
        predReviews = data.predReviews
        predLabels = data.predLabels
        lens = len(predLabels)


    print('---- data prepared ------')
    #
    wordEmbedding = data.wordEmbedding

    #labelList = data.labelList


    #embeddedPosition = fixedPositionEmbedding(lens, config.sequenceLength)

    loss0 = 10;f_beta0 = 0;no_up=0
    # 定义计算图
    print('---start graph---')
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth=True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():

            transformer = Transformer(config,wordEmbedding)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.MomentumOptimizer(learning_rate=config.training.learningRate,momentum=0.1) #
            #optimizer = tf.train.AdamOptimizer(config.training.learningRate,beta1=0.9,beta2=0.999,epsilon=1e-08,)
            #optimizer = tf.keras.optimizers.SGD(learning_rate=config.training.learningRate, momentum=0.1,nesterov=False)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.training.learningRate)
            #adadelta
            #optimizer = tf.train.RMSPropOptimizer(config.training.learningRate, decay=0.9, momentum=0.1, epsilon=1e-10,)
            
            
            gradsAndVars = optimizer.compute_gradients(transformer.loss)
            '''
            mean_grad = tf.zeros(())
            for grad, var in gradsAndVars:
                mean_grad += tf.reduce_mean(grad)
            mean_grad /= len(gradsAndVars)
            grads_and_vars = [(grad-mean_grad, var)  for grad, var in gradsAndVars]
            '''
            # 计算梯度,得到梯度和变量
            #gradsAndVars = optimizer.compute_gradients(transformer.loss)  #尝试梯度除以batch_size
            #grads_and_vars = [(tf.clip_by_norm(grad, 5), var)  for grad, var in gradsAndVars]
            # 将梯度应用到变量下，生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

            '''
            # 用summary绘制tensorBoard
            gradSummaries = []
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            print("Writing to {}\n".format(outDir))

            lossSummary = tf.summary.scalar("loss", transformer.loss)
            summaryOp = tf.summary.merge_all()

            trainSummaryDir = os.path.join(outDir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
            '''

            # 初始化所有变量
            print('---- start train ----')
            if args.action == 'train':
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) #
                
            sess.run(tf.global_variables_initializer())
            if (args.action == 'pred') or (args.train_continue == 1) :
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                if args.model_ckpt_id is None:
                    saver.restore(sess, tf.train.latest_checkpoint('./'+args.model_save_path+'/'))
                else:
                    saver.restore(sess,args.model_save_path+'/my-model-'+args.model_ckpt_id)

            def trainStep(batchX, batchY):
                """
                训练函数
                """
                feed_dict = {
                  transformer.inputX: batchX,
                  transformer.inputY: batchY,
                  transformer.dropoutKeepProb: config.model.dropoutKeepProb
                }
                '''
                _, summary, step, loss, predictions = sess.run(
                    [trainOp, summaryOp, globalStep, transformer.loss, transformer.predictions],
                    feed_dict)
                '''
                _,  step, loss, predictions = sess.run(
                    [trainOp,globalStep, transformer.loss, transformer.predictions],
                    feed_dict)

                if config.numClasses == 1:
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)


                elif config.numClasses > 1:
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                  labels=labelList)

                #trainSummaryWriter.add_summary(summary, step)

                return loss, acc, prec, recall, f_beta

            def devStep(batchX, batchY):
                """
                验证函数
                """
                feed_dict = {
                  transformer.inputX: batchX,
                  transformer.inputY: batchY,
                  transformer.dropoutKeepProb: 1.0
                }
                '''
                summary, step, loss, predictions = sess.run(
                    [summaryOp, globalStep, transformer.loss, transformer.predictions],
                    feed_dict)
                '''
                step, loss, logits,predictions = sess.run(
                    [globalStep, transformer.loss, transformer.logits,transformer.predictions],feed_dict)

                if config.numClasses == 1:
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)


                elif config.numClasses > 1:
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                  labels=labelList)

                #trainSummaryWriter.add_summary(summary, step)

                return loss, acc, prec, recall, f_beta, logits,predictions



            if args.action == 'train':
                for i in range(config.training.epoches):
                    # 训练模型
                    print("start training model")
                    for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize,args.action,config):

                        loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                        currentStep = tf.train.global_step(sess, globalStep)
                        print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            currentStep, loss, acc, recall, prec, f_beta))
                        if currentStep % config.training.evaluateEvery == 0: #
                            print("\nEvaluation:")

                            losses = []
                            accs = []
                            f_betas = []
                            precisions = []
                            recalls = []
                            pre = []

                            for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize, 'pred', config):
                                loss, acc, precision, recall, f_beta,logit,prediction= devStep(batchEval[0], batchEval[1])
                                losses.append(loss)
                                accs.append(acc)
                                f_betas.append(f_beta)
                                precisions.append(precision)
                                recalls.append(recall)
                            #acc1, recall1, prec1, f_beta1 = get_binary_metrics(pred_y=pre, true_y=evalLabels[0:len(pre)])
                                
                            time_str = datetime.datetime.now().isoformat()

                            writer = "{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(
                                                                        time_str,currentStep, mean(losses),
                                                                        mean(accs), mean(precisions),mean(recalls),mean(f_betas))
                            print(writer)

                            #if currentStep % config.training.checkpointEvery == 0:
                            if mean(losses)<loss0: #mean(f_betas)>f_beta0: #
                            # 保存模型的另一种方法，保存checkpoint文件
                                path = saver.save(sess, "./"+args.model_save_path+"/my-model-"+str(np.round(mean(f_betas),3))+'-'+str(np.round(mean(losses),2)), global_step=currentStep)
                                loss0 = mean(losses) #if mean(losses)<loss0 else loss0
                                f_beta0 = mean(f_betas)
                                no_up = 0
                                print("Saved model checkpoint to {}\n".format(path))
                                with open('./'+args.model_save_path+'/stat.txt','a') as fw:
                                    fw.write(writer+'\n')
                            else:
                                no_up += 1
                                if no_up >= config.training.no_upnum:
                                    break

                '''
                inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
                          "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}

                outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}

                prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
                builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                    signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

                builder.save()
                '''
            elif args.action == 'pred':
                print('---预测---')
                pred_path = 'result/'+args.pred_dir[-9:]+'_pred.npy'
                pre = []
                for batchpred in nextBatch(predReviews, predLabels, len(predLabels), args.action,config): #config.batchSize
                    loss, acc, precision, recall, f_beta,logit,prediction = devStep(batchpred[0], batchpred[1])
                    pre = np.append(prediction,logit)
                np.save(pred_path,pre)
