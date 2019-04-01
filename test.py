# -*- coding: utf-8 -*-
# @author: He Xiao (hexiao214370@sohu-inc.com)
# @date: 2019-03-29
from joblib import load
from train import Train

class Test(Train):
    def __init__(self):
        super(Test, self).__init__()
        self.tfIdf = load('models/tfIdf.joblib')
        self.coreEntityLR = load('models/coreEntityCLF.joblib')

    def testCoreEntity(self):
        testData = self.loadData('data/test.txt')
        for news in testData:
            tfIdfNameScore = self.getTfIdfScore(news, self.tfIdf)
            # print(tfIdfNameScore)
            print(news['title'])
            for name, score in tfIdfNameScore:
                if(self.coreEntityLR.predict([[score]]) > 0.5):
                    print(name)


if __name__ == '__main__':
    test = Test()
    test.testCoreEntity()


