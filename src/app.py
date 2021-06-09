print('1')

#from src.__init__ import main

from flask import Flask, request
from flask_restx import Api, Namespace, fields, Resource

## local
from settings import model_path, vocab_path, cnn_path
from torch import nn

from src.controller.analyzeController import Analyze
from src.controller.keywordController import Keyword
from src.controller.testController import Sample
from src.controller.divideHighlightController import Divide_Highlight


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





#from load_models import koBERT_CNN_Classifier
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



api.add_namespace(Divide_Highlight, '/api/v1/divide_highlight')
api.add_namespace(Keyword, '/api/v1/keyword')
api.add_namespace(Analyze, '/api/v1/analyze')
api.add_namespace(Sample, '/api/v1/test')



''' test '''
print('sdfsdfsdfsdf')

print('14')

print('sdfsdfwerwer')
print('sdfsdfsdf')

from load_models import koBERT_CNN_Classifier

prediction = koBERT_CNN_Classifier(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)
# app.run(debug=True, host='0.0.0.0')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

