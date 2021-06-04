import re
import pandas as pd

'''
    for simple preprocessing korean text
    list of helper function
    1. deleteKorean
    1-1. check korean text
    2. replace variable name to '변수'
    3. replace number to hangul text1
    4. replace number to hangul text2
    5. delete emoji
    6. nan preprocessing
    7. ner preprocessing : replace human name, location, organization to '이름','지역','조직'
    8. delete enter tab in description
    0.
'''


class textPreprocessor:

    @staticmethod
    def deleteSpecialCharacters(text):
        '''
        :param text: String : description
        :return: String : special characters dropped text
        '''
        non_special_text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ]', '', text)
        return non_special_text

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
    def deleteEnglish(text):
        '''
        :param text: String : description
        :return: String : english dropped text
        '''
        non_english_text = re.sub('[a-zA-z]', '', text)
        return non_english_text

    @staticmethod
    def replaceBracket(text):
        '''
        :param text: String : description
        :return: String : bracket replaced text
        '''
        characters = "(),."
        resultList = []
        for x in range(len(characters)):
            text = text.replace(characters[x], "")
        return text

    @staticmethod
    def replaceNan2Text(text):
        '''
        :param text: String : description
        :return: String : nan replaced text
        '''
        if pd.isna(text):
            text = "이 문제는 입력이 없습니다."
            return text
        else:
            return text

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

    @staticmethod
    def replaceNumber2String(text):
        '''
        :param text: String : description
        :return: String : number replaced to hangul text
        '''
        numbers = []
        numbers = re.findall("\d+", text)
        numbers.sort(key=lambda x: len(x), reverse=False)
        if numbers == None:
            return text
        for oneNumber in numbers:
            if len(oneNumber) == 1:
                text = text.replace(oneNumber, " 숫자 ")
            elif 2 <= len(oneNumber) <= 3:
                text = text.replace(oneNumber, " 작은 숫자 ")
            else:
                text = text.replace(oneNumber, " 큰 숫자 ")
        return text

    @staticmethod
    def findNumber(text):
        '''
        :param text: String : description
        :return: List : list of integer string in text
        '''
        numbers = re.findall("\d+", text)
        numbers.sort(key=lambda x: len(x), reverse=True)
        return numbers

    @staticmethod
    def readNumber(n):
        '''
        :param n: integer : one integer
        :return: String : number replaced by hangul text
        '''
        units = [''] + list('십백천')
        nums = '일이삼사오육칠팔구'
        result = []
        i = 0
        while n > 0:
            n, r = divmod(n, 10)
            if r > 0:
                result.append(nums[r - 1] + units[i])
            i += 1
        return ''.join(result[::-1])

    @staticmethod
    def readNumber2(n):
        '''
        :param n: integer under length 9
        :return: String : number replaced by hangul text
        '''
        if len(str(n)) >= 9:
            return False
        a, b = [n.readNumber(x) for x in divmod(n, 10000)]
        if a:
            return a + "만" + b
        return b

    @staticmethod
    def changeNumber2Hangul(text):
        '''
        :param text: String : description
        :return: String : number replaced by hangul text
        '''
        if text.findNumber(text) == []:
            for oneNumber in text.findNumber(text):
                temp = text.readNumber2(int(oneNumber))
                if temp is not False:
                    text = text.replace(oneNumber, text.readNumber2(int(oneNumber)))
                else:
                    text = text.replace(oneNumber, "큰 수")
        return text

    @staticmethod
    def replaceOneEngChar(text):
        '''
        :param text: String : description
        :return: String : variable name replaced by '변수' text
        '''
        for i in range(0, len(text)):
            if i == 0:
                if len(text) != 1:
                    if (text.checkEng(text[i])) and (not text.checkEng(text[i + 1])):
                        text = '변수' + text[i + 1:]
                else:
                    text = '변수'
            elif i == len(text) - 1:
                if (text.checkEng(text[i])) and (not text.checkEng(text[i - 1])):
                    text = text[:i] + '변수'
            else:
                if (text.checkEng(text[i])) and (not text.checkEng(text[i - 1])) and (not text.checkEng(text[i + 1])):
                    text = text[:i] + '변수' + text[i + 1:]

        return text

    @staticmethod
    def deleteEmoji(text):
        '''
        :param text: String : description
        :return: String : emoji dropped description
        '''
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
