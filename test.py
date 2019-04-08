# -*- coding: utf-8 -*-
# @author: He Xiao (hexiao214370@sohu-inc.com)
# @date: 2019-03-29
from joblib import load
from train import Train
import re

class Test(Train):
    def __init__(self):
        super(Test, self).__init__()
        self.coreEntityTfIdf = load('models/coreEntityTfIdf.joblib')
        self.coreEntityCLF = load('models/coreEntityCLF.joblib')

        self.emotionTfIdf = load('models/emotionTfIdf.joblib')
        self.emotionCLF = load('models/emotionCLF.joblib')

    def testCoreEntity(self):
        testData = self.loadData('data/coreEntityEmotion_test_stage1.txt')
        for news in testData:
            predictCoreEntityEmotion = {}

            tfIdfNameScore = self.getTfIdfScore(news, self.coreEntityTfIdf)

            # predict core Entities
            coreEntities = []
            for name, score in tfIdfNameScore:
                if(self.coreEntityCLF.predict([[score]]) > 0.5):
                    coreEntities.append(name)

            # predict emotion of core entity
            for entity in coreEntities:
                text = news['title'] + '\n' + news['content']
                relatedSents = []
                for sent in re.split(r'[\n\t，。！？“”（）]', text):
                    if (entity in sent):
                        relatedSents.append(sent)
                relatedText = ' '.join(relatedSents)
                emotionTfIdfFeature = self.emotionTfIdf.transform([relatedText]).toarray()
                emotion = self.emotionCLF.predict(emotionTfIdfFeature)
                predictCoreEntityEmotion[entity] = emotion[0]

            print(news['title'], predictCoreEntityEmotion)



if __name__ == '__main__':
    test = Test()
    test.testCoreEntity()


