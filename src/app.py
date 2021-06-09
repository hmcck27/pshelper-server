print('1')

#from src.__init__ import main

from flask import Flask, request
from flask_restx import Api, Namespace, fields, Resource

# from src.controller.analyzeController import Analyze
from src.controller.keywordController import Keyword
from src.controller.testController import Sample
from src.controller.divideHighlightController import Divide_Highlight

from load_models import koBERT_CNN_Classifier
from settings import model_path, cnn_path, vocab_path

from torch import nn
import torch

from src.preprocessor.textPreprocessor import textPreprocessor

print('2')

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

api.add_namespace(Divide_Highlight, '/api/v1/divide_highlight')
api.add_namespace(Keyword, '/api/v1/keyword')
api.add_namespace(Analyze, '/api/v1/analyze')
api.add_namespace(Sample, '/api/v1/test')


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
print('sdfsdfsdfsdf')

@Analyze.route('')
class AnalyzeController(Resource):

    @Analyze.expect(analyze_fields)
    @Analyze.response(201, "Success", analyze_response)
    def post(self):
        content = request.json.get('content')
        text_preprocessor = textPreprocessor()
        '''
            TO-DO
            0. preprocess text
            1. analyze the description
        '''
        preprocessed_text = text_preprocessor.preprocessing(content)
        # tag = TagAnalyzer.findTag(preprocessed_text)
        tag,ratio = prediction.predict(preprocessed_text)
        # print(content)

        return {
                   'problem_id': request.json.get('problem_id'),
                   'problem_url': "https://www.acmicpc.net/problem/" + str(request.json.get('problem_id')),
                   'algorithm_type' : tag,
                   'algorithm_ratio' : ratio
               }, 201


print('sdfsdfwerwer') 
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

    def forward(self, x) :
        out1 = self.conv1d_maxpooling1(x.transpose(1, 2))
        out2 = self.conv1d_maxpooling2(x.transpose(1, 2))
        out3 = self.conv1d_maxpooling3(x.transpose(1, 2))
        out = torch.cat((out1, out2, out3), 2)
        out = out.reshape(out.size(0), -1)
        return  self.classifier(out)    

# class Classifier(nn.Module):
#     def __init__(self,
#                  hidden_size=768,
#                  num_classes=8,
#                  dr_rate=0.0):
#         super(Classifier, self).__init__()
#         # 16, 2848
#         # 32, 5696
#         # 1312
#         self.kernel_num = 16
#         self.conv1d_maxpooling1 = nn.Sequential(
#             nn.Conv1d(hidden_size, self.kernel_num, 4, stride=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2, 1),
#             nn.Dropout(dr_rate)
#         )
#         self.conv1d_maxpooling2 = nn.Sequential(
#             nn.Conv1d(hidden_size, self.kernel_num, 8, stride=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2, 1),
#             nn.Dropout(dr_rate)
#         )
#         self.conv1d_maxpooling3 = nn.Sequential(
#             nn.Conv1d(hidden_size, self.kernel_num, 16, stride=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2, 1),
#             nn.Dropout(dr_rate)
#         )
#
#         self.classifier = nn.Linear(1312, num_classes)
#
#     def forward(self, x) :
#       out1 = self.conv1d_maxpooling1(x.transpose(1, 2))
#       out2 = self.conv1d_maxpooling2(x.transpose(1, 2))
#       out3 = self.conv1d_maxpooling3(x.transpose(1, 2))
#       out = torch.cat((out1, out2, out3), 2)
#       out = out.reshape(out.size(0), -1)
#       return  self.classifier(out)


prediction = koBERT_CNN_Classifier(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)

if __name__ == "__main__": 
    app.run(debug=True, host='0.0.0.0')

