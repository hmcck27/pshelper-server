from flask import request
from flask_restx import Resource,Namespace, fields
from src.analyzer.tagAnalyzer import TagAnalyzer
from src.preprocessor.textPreprocessor import textPreprocessor
from src.app import prediction, Analyze, analyze_response, analyze_fields
# from src.app import prediction
# from src.app import Analyze
# from src.__init__ import prediction


''' test '''
print('sdfsdfsdfsdf')

print('14')

print('sdfsdfwerwer')
@Analyze.route('')
class AnalyzeController(Resource):
    print('4444444')
    print(prediction)
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
