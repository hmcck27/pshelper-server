from src.preprocessor.deleteFunctions import deleteFunctions
from src.preprocessor.replaceFunctions import replaceFunctions
from src.preprocessor.checkFunctions import checkFunctions


class textPreprocessor():

    def __init__(self, text):
        self.preprocessedText = self.preprocessing(text)


    @classmethod
    def preprocessing(self, text):
        '''
        :param text: String :row data
        :return: String : preprocessed data
        '''
        if not checkFunctions.checkKorean(text):
            return False
        text = deleteFunctions.deleteEmoji(text)
        # text = deleteFunctions.deleteFunctions.deleteEnglish(text) ##
        text = deleteFunctions.deleteSpecialCharacters(text)
        text = replaceFunctions.replaceOneEngChar(text)
        text = replaceFunctions.replaceNumber2Hangul(text)
        text = replaceFunctions.replaceBracket(text)
        text = replaceFunctions.replaceNumber2String(text)
        return text
