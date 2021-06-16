from flask import Flask, request
from flask_restx import Api, Namespace, fields, Resource

from src.controller.testController import Sample
from src.controller.keywordController import Keyword
from src.controller.analyzeController import Analyze
# from src.controller.divideHighlightController import Divide_Highlight

import load_models

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

# api.add_namespace(Divide_Highlight, '/api/v1/divide_highlight')
api.add_namespace(Keyword, '/api/v1/keyword')
api.add_namespace(Analyze, '/api/v1/analyze')
api.add_namespace(Sample, '/api/v1/test')

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0')

