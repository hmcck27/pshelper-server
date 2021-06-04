from flask import request
from flask_restx import Resource, Namespace, fields
from src.analyzer.problem_analyzer import ProblemAnalyzer

Problem = Namespace(
    name="Problem",
    description="문제 지문을 받고 적절한 값을 리턴해줍니다.",
)

# Model 객체 생성
problem_fields = Problem.model('Problem', {
    'problem_id': fields.Integer(description='문제 번호', required=True, example="1007"),
    'content': fields.String(description='문제 지문', required=True,
                             example="평면 상에 N개의 점이 찍혀있고, 그 점을 집합 P라고 하자. 하지만 집합 P의 벡터 매칭은 벡터의 집합인데, 모든 벡터는 집합 P의 한 점에서 시작해서, 또 다른 점에서 끝나는 벡터의 집합이다. 또, P에 속하는 모든 점은 한 번씩 쓰여야 한다.V에 있는 벡터의 개수는 P에 있는 점의 절반이다.평면 상의 점이 주어졌을 때, 집합 P의 벡터 매칭에 있는 벡터의 합의 길이의 최솟값을 출력하는 프로그램을 작성하시오."),
    'input': fields.String(description='문제 입력사항', required=False,
                           example="첫째 줄에 테스트 케이스의 개수 T가 주어진다. 각 테스트 케이스는 다음과 같이 구성되어있다. 테스트 케이스의 첫째 줄에 점의 개수 N이 주어진다. N은 짝수이다. 둘째 줄부터 N개의 줄에 점의 좌표가 주어진다. N은 20보다 작거나 같은 자연수이고, 좌표는 절댓값이 100,000보다 작거나 같은 정수다. 모든 점은 서로 다르다."),
    'algorithm_types': fields.List(fields.String, description='문제 알고리즘 분류', required=False, example="")
})

problem_response = Problem.model('Problem_Response', {
    'problem_id': fields.Integer,
    'problem_url': fields.String,
    'sentence_list': fields.List(fields.String),
})


@Problem.route('')
class ProblemController(Resource):
    @Problem.expect(problem_fields)
    @Problem.response(201, 'Success', problem_response)
    # @Problem.marshal_with(problem_fields)
    def post(self):
        content = request.json.get('content')
        sentence_list = ProblemAnalyzer.getDevidedContent(content)
        return {
                   'problem_id': request.json.get('problem_id'),
                   'problem_url': "https://www.acmicpc.net/problem/" + str(request.json.get('problem_id')),
                   'sentence_list': sentence_list
               }, 201

    # @Problem.expect(problem_fields)
    # @Problem.response(201, 'Success', problem_fields)
    # def get(self):
    #     return {
    #                'problem_id': 0,
    #                'data': 'test'
    #            }, 201
