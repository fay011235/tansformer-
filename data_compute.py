# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:16:55 2020

@author: fay
"""
import json

import warnings
from collections import Counter
import gensim
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")
# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource        
        self._predSource = config.predSource
        self._devSource = config.devSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding =None

        self.labelList = []

    def _readData(self, filePath,action):
        """
        从csv文件中读取数据集
        """

        df = pd.read_csv(filePath)
        if action != 'pred':
            df = shuffle(df)

        if self.config.numClasses == 1:
            labels = df["result"].tolist()
        elif self.config.numClasses > 1:
            labels = df["rate"].tolist()

        review = df["time_inter_warn"].tolist() 
        reviews = [line.strip().split(' ') for line in review]

        return reviews, labels

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[str(label)] for label in labels]
        return labelIds

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for i in range(len(x)):
            review = x[i]
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
        '''
        for i,review in enumerate(x):
            position = posits[i]
            if (len(review) >= self._sequenceLength):
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

            if (len(position) >= self._sequenceLength):
                posit = position[:self._sequenceLength]
            else:
                #print('----position----:',position)
                #print('----type,position----:',type(position))
                posit = position + [193] * (self._sequenceLength - len(position))
                #print('----len,position----:',len(position))
            #print('----posit----:',posit)
            #print('----len2,position----:',len(posit))
            #posit_emb = fixedPositionEmbedding(posit,self._sequenceLength)
            positions.append(posit)
        print('---len(positions):---',len(positions))
        '''
        trainIndex = int(len(x) * rate)
        #print('----reviews type ----:',reviews)
        trainReviews = np.array(reviews[:trainIndex], dtype="float32")
        trainLabels = np.array(y[:trainIndex], dtype="int64")

        evalReviews = np.array(reviews[trainIndex:], dtype="float32")
        evalLabels = np.array(y[trainIndex:], dtype="int64")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genPredData(self, x, y, word2idx):
        """
        生成预测集
        """
        reviews = []
        '''
        for i,review in enumerate(x):
            position = posits[i]
            if (len(review) >= self._sequenceLength):
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

            if (len(position) >= self._sequenceLength):
                posit = position[:self._sequenceLength]
            else:
                posit = position + [193] * (self._sequenceLength - len(position))
            positions.append(posit)
        '''

        for i in range(len(x)):
            review = x[i]
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        predReviews = np.asarray(reviews, dtype="int64")
        predLabels = np.array(y, dtype="float32")

        return predReviews,predLabels

    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        if not os.path.exists('data/wordJson/label2idx.json'):

            allWords = [word for review in reviews for word in review]

            # 去掉停用词
            subWords = [word for word in allWords if word not in self.stopWordDict]

            wordCount = Counter(subWords)  # 统计词频
            sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

            # 去除低频词
            words = [item[0] for item in sortWordCount if item[1] >= 2] #

            vocab, wordEmbedding = self._getWordEmbedding(words)
            self.wordEmbedding = wordEmbedding

            word2idx = dict(zip(vocab, list(range(len(vocab)))))

            uniqueLabel = list(set(labels))
            label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
            self.labelList = list(range(len(uniqueLabel)))

            # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
            with open("data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
                json.dump(word2idx, f)

            with open("data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
                json.dump(label2idx, f)
            np.save('data/wordJson/embeding.npy',wordEmbedding)

        else:
            with open('data/wordJson/word2idx.json') as f_obj1:
                word2idx = json.load(f_obj1)
            with open('data/wordJson/label2idx.json') as f_obj2:
                label2idx = json.load(f_obj2)
            self.wordEmbedding = np.load('data/wordJson/embeding.npy')
        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2Vec_time_inter_posit_60.bin", binary=False)
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vocab.append(word)
                vector = wordVec.wv[word]
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """

        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self,action):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        # 初始化数据集
        if action == 'train':
            file_path = self._dataSource
        elif action == 'pred':
            file_path = self._predSource
        reviews, labels = self._readData(file_path,action)

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)

        #计算wordEmbeding
        '''
        path = './word2vec/wordEmbeding.npy'
        print('-----start wordembeding-----')
        if not os.path.exists(path):
            #reviews0, labels0, positions = self._readData(self._dataSource)
            #_, _ = self._genVocabulary(reviews0, labels0)
            np.save(path,self.wordEmbedding)#pickle.dump(self.wordEmbedding,path)
        else:
            self.wordEmbedding = np.load(path)
        print('-----end wordembeding-----')
        '''
        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)

        if action == 'train':
            # 初始化训练集和测试集
            trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds,labelIds, word2idx, 1)
            print('-----gettrainevaldata--------')
            self.trainReviews = trainReviews
            self.trainLabels = trainLabels

            reviews1, labels1 = self._readData(self._devSource,action)
            labelIds1 = self._labelToIndex(labels1, label2idx)
            reviewIds1 = self._wordToIndex(reviews1, word2idx)
            self.evalReviews,self.evalLabels = self._genPredData(
                                  reviewIds1, labelIds1, word2idx)

        if (action == 'pred'):
            predReviews,predLabels = self._genPredData(reviewIds, labelIds, word2idx)
            self.predReviews = predReviews
            self.predLabels = predLabels
# 输出batch数据集