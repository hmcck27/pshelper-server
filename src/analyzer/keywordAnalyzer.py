'''
    this is for keyword tagging
'''


class KeywordAnalyzer:
    def __init__(self):

        '''
            TO-DO -> dictionary 정의 필요
        '''

        self.keyword_dict = {}
        self.keyword = [{"label": "mathmatic", "keywords" : []}]
        self.tag = {}
        ## return-value

    def findKeyword(self,text):
        for self.keyword["keywords"] in self.keyword :
            keywordList = self.keyword["keywords"]
            # for oneKeyword in keywordList :


