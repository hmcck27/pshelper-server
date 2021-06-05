import pandas as pd
import re
from src.preprocessor.checkFunctions import checkFunctions


def findNumber(text):
    '''
    :param text: String : description
    :return: List : list of integer string in text
    '''
    numbers = re.findall("\d+", text)
    numbers.sort(key=lambda x: len(x), reverse=True)
    return numbers


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


class replaceFunctions:

    @staticmethod
    def replaceBracket(text):
        '''
        :param text: String : description
        :return: String : bracket replaced text
        '''
        characters = "(),."
        resultList = []
        for x in range(len(characters)):
            text = text.replace(characters[x], " ")
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
    def replaceNumber2String(text):
        '''
        :param text: String : description
        :return: String : number replaced to hangul text
        '''
        numbers = []
        numbers = re.findall("\d+", text)
        numbers.sort(key=lambda x: len(x), reverse=False)
        if numbers is None:
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
    def replaceNumber2Hangul(text):
        '''
        :param text: String : description
        :return: String : number replaced by hangul text
        '''
        if not findNumber(text):
            for oneNumber in findNumber(text):
                temp = readNumber2(int(oneNumber))
                if temp is not False:
                    text = text.replace(oneNumber, readNumber2(int(oneNumber)))
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
                    if (checkFunctions.checkEng(text[i])) and (not checkFunctions.checkEng(text[i + 1])):
                        text = '변수' + text[i + 1:]
                else:
                    text = '변수'
            elif i == len(text) - 1:
                if (checkFunctions.checkEng(text[i])) and (not checkFunctions.checkEng(text[i - 1])):
                    text = text[:i] + '변수'
            else:
                if (checkFunctions.checkEng(text[i])) and (not checkFunctions.checkEng(text[i - 1])) and (
                not checkFunctions.checkEng(text[i + 1])):
                    text = text[:i] + '변수' + text[i + 1:]
        return text
