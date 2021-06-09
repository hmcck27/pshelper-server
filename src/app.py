# from src.__init__ import main


#from src.__init__ import main

from flask import Flask, request

from flask_restx import Api, Namespace, fields, Resource

import torch
from torch import nn
import gluonnlp as nlp
from transformers import BertModel

from load_models import koBERT_CNN_Classifier
## local
from settings import model_path, vocab_path, cnn_path
from torch import nn

# from src.controller.analyzeController import Analyze
print('2')
from src.controller.keywordController import Keyword
from src.controller.testController import Sample
from src.controller.divideHighlightController import Divide_Highlight

#from load_models import koBERT_CNN_Classifier
from settings import model_path, cnn_path, vocab_path
print('3')
from torch import nn
import torch

from src.preprocessor.textPreprocessor import textPreprocessor



app = Flask(__name__)
api = Api(
    app,
    version='0.1',
    title="PS HELPER API Server",
    description="PS HELPER API 문서입니다.",
    terms_url="/",
    contact_url="donghoon149@gmail.com / hmcck27@gmail.com",
    license="MIT"
)

Analyze = Namespace(
    name="Analyze Algorithm",
    description='문제 지문을 받고 적절한 <strong>알고리즘 태그</strong>를 반환합니다.',
)

# Model 객체 생성
analyze_fields = Analyze.model('Problem', {
    'problem_id': fields.Integer(description='문제 번호', required=True, example="1007"),
    'content': fields.String(description='문제 지문', required=True,
                             example="평면 상에 N개의 점이 찍혀있고, 그 점을 집합 P라고 하자. 하지만 집합 P의 벡터 매칭은 벡터의 집합인데, 모든 벡터는 집합 P의 한 점에서 시작해서, 또 다른 점에서 끝나는 벡터의 집합이다. 또, P에 속하는 모든 점은 한 번씩 쓰여야 한다.V에 있는 벡터의 개수는 P에 있는 점의 절반이다.평면 상의 점이 주어졌을 때, 집합 P의 벡터 매칭에 있는 벡터의 합의 길이의 최솟값을 출력하는 프로그램을 작성하시오."),
    'input': fields.String(description='문제 입력사항', required=False,
                           example="첫째 줄에 테스트 케이스의 개수 T가 주어진다. 각 테스트 케이스는 다음과 같이 구성되어있다. 테스트 케이스의 첫째 줄에 점의 개수 N이 주어진다. N은 짝수이다. 둘째 줄부터 N개의 줄에 점의 좌표가 주어진다. N은 20보다 작거나 같은 자연수이고, 좌표는 절댓값이 100,000보다 작거나 같은 정수다. 모든 점은 서로 다르다."),
})

algorithm_fields = fields.Wildcard(fields.String)

analyze_response = Analyze.model('Problem_response', {
    'problem_id': fields.String(description='문제 번호', required=True, example="1007"),
    'problem_url': fields.String(description="문제 url", required=True, example="www.psHelper.de"),
    'algorithm_type': algorithm_fields
})
''' test '''

tag_name = ['Mathematics', 'Dynamic_programming', 'Implementation', 'Graph_theory',
       'Data_structures', 'Greedy', 'String', 'Graph_traversal',
       'Bruteforcing', 'Tree', 'Binary_search', 'Number_theory',
       'Breadth_first_search', 'Depth_first_search', 'Dijkstras',
       'Divide_and_conquer', 'Stack', 'Priority_queue']

thresholds = [0.4, 0.34, 0.3, 0.35,
              0.25, 0.16, 0.28, 0.27,
              0.14, 0.18, 0.14, 0.28,
              0.19, 0.13, 0.2,
              0.1, 0.05, 0.06]
class BERToutput(nn.Module):
    def __init__(self,
                 bert):
        super(BERToutput, self).__init__()
        self.bert = bert

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        token_ids = torch.from_numpy(token_ids.reshape(1, -1))
        valid_length = torch.from_numpy(valid_length.reshape(1, -1))
        segment_ids = torch.from_numpy(segment_ids.reshape(1, -1))

        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        op, _ = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float(),
                          return_dict=False)
        x = op[:, :16, :]
        y = op[:, -48:, :]
        output = torch.cat((x, y), dim=1)

        return output


class Classifier(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_classes=8,
                 dr_rate=0.0):
        super(Classifier, self).__init__()
        # 16, 2848
        # 32, 5696
        # 1312
        self.kernel_num = 16
        self.conv1d_maxpooling1 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 4, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )
        self.conv1d_maxpooling2 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )
        self.conv1d_maxpooling3 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 16, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )

        self.classifier = nn.Linear(1312, num_classes)

    def forward(self, x):
        out1 = self.conv1d_maxpooling1(x.transpose(1, 2))
        out2 = self.conv1d_maxpooling2(x.transpose(1, 2))
        out3 = self.conv1d_maxpooling3(x.transpose(1, 2))
        out = torch.cat((out1, out2, out3), 2)
        out = out.reshape(out.size(0), -1)

        return self.classifier(out)


def get_kobert_classifier(model_path, vocab_path, cnn_path):
    bert = BertModel.from_pretrained(model_path)
    bertmodel = BERToutput(bert)
    bertmodel.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path, padding_token='[PAD]')
    tok = nlp.data.BERTSPTokenizer(vocab_path, vocab, lower=False)
    classifier = torch.load(cnn_path, map_location=torch.device('cpu'))
    classifier.eval()
    return bertmodel, tok, classifier


def convert_to_vector(bertmodel, tok, text) :
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=512, pad=True, pair=False)
    sentences = transform([text])
    token_ids, valid_length, segment_ids = sentences
    vector = bertmodel(token_ids, valid_length, segment_ids)
    return vector

def classification(classifier, vector, thresholds) :
  out = classifier(vector)
  out = torch.sigmoid(out).squeeze(0).tolist()
  label = [tag_name[i] for i in range(len(out))if out[i] >= thresholds[i]]
  return label



print('sdfsdfwerwer')
api.add_namespace(Divide_Highlight, '/api/v1/divide_highlight')
api.add_namespace(Keyword, '/api/v1/keyword')
api.add_namespace(Analyze, '/api/v1/analyze')
api.add_namespace(Sample, '/api/v1/test')


@Analyze.route('')
class AnalyzeController(Resource):
    print('4444444')
    @Analyze.expect(analyze_fields)
    @Analyze.response(201, "Success", analyze_response)
    def post(self):
        print('sdfsdfsdfsdfsdfsdfsdfsdf')
        content = request.json.get('content')
        text_preprocessor = textPreprocessor()
        '''
            TO-DO
            0. preprocess text
            1. analyze the description
        '''
        preprocessed_text = text_preprocessor.preprocessing(content)

        # tag, ratio = model.predict(preprocessed_text)

        return {
                   'problem_id': request.json.get('problem_id'),
                   'problem_url': "https://www.acmicpc.net/problem/" + str(request.json.get('problem_id')),
                   'algorithm_type' : tag,
                   'algorithm_ratio' : ratio
               }, 201





''' test '''
print('sdfsdfsdfsdf')

print('14')

print('sdfsdfwerwer')
print('sdfsdfsdf')
text = "칠개의 자연수가 주어질 때, 이들 중 홀수인 자연수들을 모두 골라 그 합을 구하고, 고른 홀수들 중 최솟값을 찾는 프로그램을 작성하시오.예를 들어, 칠개의 자연수 일십이, 칠십칠, 삼십팔, 사십일, 오십삼, 구십이, 팔십오가 주어지면 이들 중 홀수는 칠십칠, 사십일, 오십삼, 팔십오이므로 그 합은칠십칠 + 사십일 + 오십삼 + 팔십오 = 이백오십육이 되고,사십일 < 오십삼 < 칠십칠 < 팔십오이므로 홀수들 중 최솟값은 사십일이 된다."
# model = koBERT_CNN_Classifier()
# model.initModel(cnn_path=cnn_path, vocab_path=vocab_path, model_path=model_path)

# app.run(debug=True, host='0.0.0.0')

if __name__ == "__main__":
    bert, tokenizer, classifier = get_kobert_classifier(model_path, vocab_path, cnn_path)
    vector = convert_to_vector(bert, tokenizer, text)
    out = classification(classifier, vector, thresholds)
    app.run(debug=True, host='0.0.0.0')

