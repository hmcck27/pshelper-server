from flask import request
from flask_restx import Resource, Namespace, fields

Sample = Namespace(
    name="test",
    description="테스트용 api입니다.",
)

@Sample.route('')
class textController(Resource):
    def get(self):
        return "hello world";

