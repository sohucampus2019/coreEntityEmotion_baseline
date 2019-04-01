# -*- coding: utf-8 -*-

import jieba
import codecs
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from joblib import dump
import re

class Train():
    def __init__(self):
        # load nerDict as named entity recognizer
        self.loadNerDict()

    def trainCoreEntity(self):
        '''
        train model for coreEntity
        Baseline use entityDict for named entity recognition, you can use a more wise method.
        Baseline use tfidf-score as feature, LR as classification model
        :return:
        '''
        # 1. train tfIdf as core entity score model
        trainData = self.loadData('data/train.txt')

        nerCorpus = []
        for news in trainData:
            nerCorpus.append(' '.join(self.getEntity(news)))


        tfIdf = TfidfVectorizer()
        tfIdf.fit(nerCorpus)
        # 1.1 save tfIdf model
        dump(tfIdf, 'models/tfIdf.joblib')

        # 2. train LR with tfIdf score as features
        isCoreX = []
        isCoreY = []
        for news in trainData:

            tfIdfNameScore = self.getTfIdfScore(news, tfIdf)

            coreEntity_GroundTruth = [x['entity'] for x in news['coreEntityEmotions']]
            for name, score in tfIdfNameScore:
                if(name in coreEntity_GroundTruth):
                    isCoreX.append([score])
                    isCoreY.append(1)
                else:
                    isCoreX.append([score])
                    isCoreY.append(0)

        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial').fit(isCoreX, isCoreY)
        dump(clf, 'models/CoreEntityCLF.joblib')

    def trainEmotion(self):
        '''
        train emotion model
        Baseline use tfidf vector as feature, NaiveBayes as classfication model
        :return:
        '''
        trainData = self.loadData('data/train.txt')

        emotionX = []
        emotionY = []

        for news in trainData:

            text = news['title'] + '\n' + news['content']
            entities = [x['entity'] for x in news['coreEntityEmotions']]
            emotions = [x['emotion'] for x in news['coreEntityEmotions']]
            entityEmotionMap = dict(zip(entities, emotions))
            entitySentsMap = {}
            for entity in entityEmotionMap.keys():
                entitySentsMap[entity] = []

            for sent in re.split(r'[\n\t，。！？“”（）]',text):
                for entity in entityEmotionMap.keys():
                    if(entity in sent):
                        entitySentsMap[entity].append(sent)

            for entity, sents in entitySentsMap.items():
                relatedText = ' '.join(sents)
                emotionX.append(relatedText)
                emotionY.append(entityEmotionMap[entity])


        clf = GaussianNB()






    def getTfIdfScore(self, news, tfIdf):
        featureName = tfIdf.get_feature_names()

        doc = self.getEntity(news)

        tfIdfFeatures = tfIdf.transform([' '.join(doc)])

        tfIdfScores = tfIdfFeatures.data
        # normalize
        tfIdfScoresNorm = normalize([tfIdfScores], norm='max')

        tfIdfNameScore = [(featureName[x[0]], x[1]) for x in zip(tfIdfFeatures.indices, tfIdfScoresNorm[0])]
        tfIdfNameScore = sorted(tfIdfNameScore, key=lambda x: x[1], reverse=True)

        return tfIdfNameScore

    def loadNerDict(self):
        nerDictFile = codecs.open('models/nerDict.txt','r','utf-8')
        self.nerDict = []
        for line in nerDictFile:
            self.nerDict.append(line.strip())

    def getWords(self, news):
        '''
        get all word list from news
        :param news:
        :return:
        '''
        title = news['title']
        content = news['content']

        words = jieba.cut(title + ' ' + content)

        return list(words)

    def getEntity(self, news):
        '''
        get all entity list from news
        :param news:
        :return:
        '''
        ners = []
        words = self.getWords(news)
        for word in words:
            if (word in self.nerDict):
                ners.append(word)
        return ners

    def loadData(self, filePath):
        f = codecs.open(filePath,'r', 'utf-8')
        data = []
        for line in list(f.readlines())[0:10]:
            news = json.loads(line.strip())
            data.append(news)
        return data



if __name__ == '__main__':
    trainer = Train()
    trainer.trainEmotion()
