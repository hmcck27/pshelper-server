from src.resource.keyword import keyword_list
from src.resource.keyword import label_keyword_ratio
'''
    this is for keyword tagging
'''

class KeywordAnalyzer:
    def __init__(self,text):

        '''
            TO-DO -> dictionary 정의 필요
        '''

        self.keyword_dict = {}
        self.keyword_dict["keyword"] = {}
        self.keyword = keyword_list
        self.keyword = label_keyword_count
        self.highlighted_text=''
        self.findKeyword(text)
        ## return-value

    def findKeyword(text):
        for oneKeyword in self.keyword :
            index = text.find(oneKeyword["name"])
            if index != -1 :
                if text[index-1] != ' ' :
                    continue
                else :
                    text = text.replace(oneKeyword["name"],"<mark>" + oneKeyword["name"]  + "</mark>")
                    temp_dict = {}
                    wholeCount = sum([int(x) for x in oneKeyword["count"].split(',') ])
                    for one_label, one_count in zip(oneKeyword["label"].split(','), oneKeyword["count"].split(',')) :
                        temp_dict[one_label] = str(int((int(one_count) * 100) / wholeCount)) + '%'
                    self.keyword_dict["keyword"][oneKeyword["name"]] = temp_dict

        self.highlighted_text = text

class keywordAnalyzer2:

    def findKeyword(text):
        keyword_dict = {}
        for oneKeyword in keyword_list :
            index = text.find(oneKeyword["name"])
            if index != -1 :
                if text[index - 1] != ' ' :
                    continue
                else :
                    text = text.replace(oneKeyword["name"], "<mark>" + oneKeyword["name"] + "</mark>")
                    temp_dict = {}
                    wholeCount = sum([int(x) for x in oneKeyword["count"].split(',')[1:] ])
                    for one_label, one_count in zip(oneKeyword["label"].split(',')[1:], oneKeyword["count"].split(',')[1:]) :

                        temp_dict[one_label] = str(int((int(one_count) * 100) * label_keyword_ratio[one_label])) + '%'

                    keyword_dict["keyword"][oneKeyword["name"]] = temp_dict

        return keyword_dict, text
