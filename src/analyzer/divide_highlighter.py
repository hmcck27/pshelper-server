# from koalanlp.Util import initialize, finalize
# from koalanlp.proc import SentenceSplitter, Tagger
# from koalanlp import API
from konlpy.tag import Kkma


# initialize(OKT='LATEST')  #: HNN=2.0.3
# splitter = SentenceSplitter(API.OKT)
# tagger = Tagger(API.EUNJEON)

kkma = Kkma()

class DivideHighlighter():
    def __init__(self,text):
        self.result = []
        splitted_text = self.getDevidedContent(text)

        for one_sentence in splitted_text:
            bolded_sentence = self.getPosBySentence(one_sentence)
            self.result.append(bolded_sentence)

    @classmethod
    def getPosBySentence(self,sentence):
        posses = kkma.pos(sentence)
        pos_dict = {}
        sentence_formatted = sentence
        for pos in posses:
            if pos[1] == 'MAC':
                pos_dict[pos[0]] = pos[1]
        for key in pos_dict:
            sentence_formatted = sentence_formatted.replace(key, "<strong>"+key+"</strong>")
        print(sentence_formatted)
        return sentence_formatted

    @classmethod
    def getDevidedContent(self,content):
        sentences = kkma.sentences(content)
        print(sentences)
        sentence_list = sentences
        return sentence_list


