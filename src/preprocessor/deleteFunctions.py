import re

class deleteFunctions:
    @staticmethod
    def deleteSpecialCharacters(text):
        '''
        :param text: String : description
        :return: String : special characters dropped text
        '''
        non_special_text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ]', '', text)
        return non_special_text

    @staticmethod
    def deleteEnglish(text):
        '''
        :param text: String : description
        :return: String : english dropped text
        '''
        non_english_text = re.sub('[a-zA-z]', '', text)
        return non_english_text

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