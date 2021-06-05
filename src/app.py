from flask import Flask
from flask_restx import Api
from .controller.divideHighlightController import Divide_Highlight
from .controller.keywordController import Keyword
from .controller.analyzeController import Analyze
from .controller.testController import Sample

app = Flask(__name__)
api = Api(
    app,
    version='0.1',
    title="PS HELPER API Server",
    description="PS HELPER API 문서입니다.",
    terms_url="/",
    contact="donghoon149@gmail.com / hmcck27@gmail.com",
    license="MIT"
)

api.add_namespace(Divide_Highlight, '/api/v1/divide_highlight')
api.add_namespace(Keyword, '/api/v1/keyword')
api.add_namespace(Analyze, '/api/v1/analyze')
api.add_namespace(Sample, '/v1/api/test')
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)