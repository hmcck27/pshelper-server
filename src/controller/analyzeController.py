from flask import request
from flask_restx import Resource, Namespace, fields
from src.analyzer.tagAnalyzer import TagAnalyzer
from src.preprocessor.textPreprocessor import textPreprocessor

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


@Analyze.route('')
class AnalyzeController(Resource):

    @Analyze.expect(analyze_fields)
    @Analyze.response(201, "Success", analyze_response)
    def post(self):
        content = request.json.get('content')
        text_preprocessor = textPreprocessor(content)
        '''
            TO-DO
            0. preprocess text
            1. analyze the description
        '''
        content = text_preprocessor.preprocessedText
        # tag = TagAnalyzer.fineTag(content)
        print(content)

        return {
                   'problem_id': request.json.get('problem_id'),
                   'problem_url': "https://www.acmicpc.net/problem/" + str(request.json.get('problem_id')),
                   # 'algoritym_type': tag
               }, 201
