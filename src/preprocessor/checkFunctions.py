import re

class checkFunctions:
    @staticmethod
    def checkKorean(text):
        '''
        :param text: String : description
        :return: Boolean : whether it is not english only text
        '''
        hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text))
        if hanCount > 0:
            return True
        else:
            return False

    @staticmethod
    def checkEng(char):
        '''
        :param char: character : one letter
        :return: Boolean : whether it is english or not
        '''
        reg = re.compile(r'[a-zA-Z]')
        if reg.match(char):
            return True
        else:
            return False