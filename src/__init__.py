# # from src.analyzer.divide_highlighter import DivideHighlighter
# # from src.analyzer.keywordAnalyzer import KeywordAnalyzer
# # from src.analyzer.tagAnalyzer import TagAnalyzer
# #
# # from src.controller.divideHighlightController import DivideHighlightController, Divide_Highlight
# # from src.controller.keywordController import KeywordAnalyzer, Keyword
# # from src.controller.analyzeController import AnalyzeController, Analyze
# # from src.controller.testController import TestController,
#
# # prediction = koBERT_CNN_Classifier(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)
# # prediction = koBERT_CNN_Classifier(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)
#
# import torch
# from torch import nn
# import gluonnlp as nlp
# from transformers import BertModel
# from settings import model_path, vocab_path, cnn_path
# # prediction = koBERT_CNN_Classifier(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)
# print('sdfsdfsdfsdfsdfsdfsdfsdfsdfsdf')
# class BERToutput(nn.Module):
#     def __init__(self,
#                  bert):
#         super(BERToutput, self).__init__()
#         self.bert = bert
#
#     def gen_attention_mask(self, token_ids, valid_length):
#         attention_mask = torch.zeros_like(token_ids)
#         for i, v in enumerate(valid_length):
#             attention_mask[i][:v] = 1
#         return attention_mask.float()
#
#     def forward(self, token_ids, valid_length, segment_ids):
#         token_ids = torch.from_numpy(token_ids.reshape(1, -1))
#         valid_length = torch.from_numpy(valid_length.reshape(1, -1))
#         segment_ids = torch.from_numpy(segment_ids.reshape(1, -1))
#
#         attention_mask = self.gen_attention_mask(token_ids, valid_length)
#
#         op, _ = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float(),
#                           return_dict=False)
#         x = op[:, :16, :]
#         y = op[:, -48:, :]
#         output = torch.cat((x, y), dim=1)
#
#         return output
#
# if __name__ == "__main__" :
#     class koBERT_CNN_Classifier():
#
#         def __init__(self):
#             print("start")
#
#         @classmethod
#         def initModel(self, model_path, vocab_path, cnn_path):
#             '''
#             :param model_path: String : path for pretrained koBERT model
#             :param vocab_path: String : path for vocab which is needed in tokenizing
#             :param cnn_path: String : path for fine tuning model
#             '''
#
#             ''' load pretrained ert model '''
#             bert = BertModel.from_pretrained(model_path)
#             self.bertmodel = BERToutput(bert)
#             self.bertmodel.eval()
#
#             ''' load vocab '''
#             self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path, padding_token='[PAD]')
#             self.tok = nlp.data.BERTSPTokenizer(vocab_path, self.vocab, lower=False)
#
#             ''' load classifier '''
#             # self.classifier = torch.load(cnn_path, map_location=torch.device('cpu'))
#             # self.classifier.eval()
#
#             self.classifier = Classifier(num_classes=18, dr_rate=0.1)
#
#             self.classifier.load_state_dict(torch.load(cnn_path))
#
#             ''' tag list '''
#             self.tag_name = ['Mathematics', 'Dynamic_programming', 'Implementation', 'Graph_theory',
#                              'Data_structures', 'Greedy', 'String', 'Graph_traversal',
#                              'Bruteforcing', 'Tree', 'Binary_search', 'Number_theory',
#                              'Breadth_first_search', 'Depth_first_search', 'Dijkstras',
#                              'Divide_and_conquer', 'Stack', 'Priority_queue']
#
#             ''' threshold for each tags '''
#             self.thresholds = [0.4, 0.34, 0.3, 0.35,
#                                0.25, 0.16, 0.28, 0.27,
#                                0.14, 0.18, 0.14, 0.28,
#                                0.19, 0.13, 0.2,
#                                0.1, 0.05, 0.06]
#
#             # load tokenizer for input text
#             self.transform = nlp.data.BERTSentenceTransform(self.tok, max_seq_length=512, pad=True, pair=False)
#
#         def convert_to_vector(self, text):
#             sentences = self.transform([text])
#             token_ids, valid_length, segment_ids = sentences
#             vector = self.bertmodel(token_ids, valid_length, segment_ids)
#             return vector
#
#         def classification(self, vector):
#             out = self.classifier(vector)
#             out = torch.sigmoid(out).squeeze(0).tolist()
#
#             label = [self.tag_name[i] for i in range(len(out)) if out[i] >= self.thresholds[i]]
#             ratio_dict = {}
#             for output, thres, tag in zip(out, self.thresholds, self.tag_name):
#                 ratio_dict[tag] = round(output / thres, 3)
#             sorted_ratio = sorted(ratio_dict.items(),
#                                   reverse=True,
#                                   key=lambda item: item[1])
#             sorted_ratio_dict = {}
#             for label_prediction_pair in sorted_ratio[:3]:
#                 sorted_ratio_dict[label_prediction_pair[0]] = label_prediction_pair[1]
#             return label, sorted_ratio_dict
#
#         def predict(self, text):
#             return self.classification(self.convert_to_vector(text))
#
#     class Classifier(nn.Module):
#         def __init__(self,
#                      hidden_size=768,
#                      num_classes=8,
#                      dr_rate=0.0):
#             super(Classifier, self).__init__()
#             # 16, 2848
#             # 32, 5696
#             # 1312
#             self.kernel_num = 16
#             self.conv1d_maxpooling1 = nn.Sequential(
#                 nn.Conv1d(hidden_size, self.kernel_num, 4, stride=2),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2, 1),
#                 nn.Dropout(dr_rate)
#             )
#             self.conv1d_maxpooling2 = nn.Sequential(
#                 nn.Conv1d(hidden_size, self.kernel_num, 8, stride=2),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2, 1),
#                 nn.Dropout(dr_rate)
#             )
#             self.conv1d_maxpooling3 = nn.Sequential(
#                 nn.Conv1d(hidden_size, self.kernel_num, 16, stride=2),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2, 1),
#                 nn.Dropout(dr_rate)
#             )
#
#             self.classifier = nn.Linear(1312, num_classes)
#
#         def forward(self, x):
#             out1 = self.conv1d_maxpooling1(x.transpose(1, 2))
#             out2 = self.conv1d_maxpooling2(x.transpose(1, 2))
#             out3 = self.conv1d_maxpooling3(x.transpose(1, 2))
#             out = torch.cat((out1, out2, out3), 2)
#             out = out.reshape(out.size(0), -1)
#             return self.classifier(out)
#
# model = koBERT_CNN_Classifier()
# model.initModel(model_path=model_path, vocab_path=vocab_path, cnn_path=cnn_path)
