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
        self.highlighted_text=''
        self.findKeyword(text)
        ## return-value

    def findKeyword(self,text):
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
        keyword_dict["keyword"] = {}
        for oneKeyword in keyword_list :
            index = text.find(oneKeyword["name"])
            if index != -1 :
                if text[index - 1] != ' ' :
                    continue
                else :
                    text = text.replace(oneKeyword["name"], "<mark>" + oneKeyword["name"] + "</mark>")
                    temp_dict = {}
                    wholeCount = sum([int(x) for x in oneKeyword["count"].split(',')[:]])
                    for one_label, one_count in zip(oneKeyword["label"].split(',')[:], oneKeyword["count"].split(',')[:]) :

                        temp_dict[one_label] = str(float(float(one_count) * label_keyword_ratio[one_label])) + '%'

                    temp = sum([float(x[:-1]) for x in temp_dict.values()])
                    for key in temp_dict :
                        temp_dict[key] = str(round(float(temp_dict[key][:-1])/temp*100,3)) + '%'
                    # print(temp)
                    keyword_dict["keyword"][oneKeyword["name"]] = temp_dict

        return keyword_dict, text
