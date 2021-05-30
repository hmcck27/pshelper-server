from flask import Flask
from flask_restx import Api
from .controller.problem import Problem
app = Flask(__name__)
api = Api(
    app,
    version='0.1',
    title="PS HELPER API Server",
    description="PS HELPER API 문서입니다.",
    terms_url="/",
    contact="donghoon149@gmail.com",
    license="MIT"
)

api.add_namespace(Problem, '/api/v1/problem')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)