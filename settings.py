''' dir path for koBERT+CNN model inference in jk-local'''
# model_path = '/Users/backend/PycharmProjects/pshelper-server/model/kobert_from_pretrained'
# cnn_path = '/Users/backend/PycharmProjects/pshelper-server/model/freeze_cnn.pt'
# cnn_path = '/Users/backend/PycharmProjects/pshelper-server/model/cnn_dict.pt'
# vocab_path = '/Users/backend/PycharmProjects/pshelper-server/model/kobert_news_wiki_ko_cased-1087f8699e.spiece'

''' dir path for koBERT+CNN model inference in pshelper-server'''
# model_path = '/home/ubuntu/pshelper-server/model/kobert_from_pretrained'
# cnn_path = '/home/ubuntu/pshelper-server/model/freeze_cnn.pt'
# vocab_path = '/home/ubuntu/pshelper-server/model/kobert_news_wiki_ko_cased-1087f8699e.spiece'

model_path = 'C:/Users/hmcck/PycharmProjects/pythonProject/pshelper-server/model/kobert_from_pretrained'
cnn_path = 'C:/Users/hmcck/PycharmProjects/pythonProject//pshelper-server/model/freeze_cnn.pt'
cnn_path = 'C:/Users/hmcck/PycharmProjects/pythonProject//pshelper-server/model/cnn_weight.h5'
# cnn_path = '/Users/backend/PycharmProjects/pshelper-server/model/cnn_dict.pt'
vocab_path = 'C:/Users/hmcck/PycharmProjects/pythonProject//pshelper-server/model/kobert_news_wiki_ko_cased-1087f8699e.spiece'


import os
print(os.getcwd())