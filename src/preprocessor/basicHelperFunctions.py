import deleteFunctions, replaceFunctions, checkFunctions

'''
    for preprocessing korean text
'''

class textPreprocessor(deleteFunctions, replaceFunctions, checkFunctions):
    def __init__(self):
        self.preprocessedText = ''

    @classmethod
    def preprocessing(self, text):
        '''
        :param text: String :row data
        :return: String : preprocessed data
        '''
        self.preprocessedText = ''


