'''
    this is for model inference
'''

# from src.app import prediction

class TagAnalyzer:

    def findTag(text):
        '''

        :param text: String
        :return: String : tag

        '''

        '''
            TO-DO
            0. embedding through koBERT
            1. model inference
            2. get the softmax result, get thresholded result
            3. return
        '''
        labels = prediction.predict(text)

        return labels;

