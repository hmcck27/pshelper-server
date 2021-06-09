import torch
from torch import nn
import gluonnlp as nlp
import numpy as np
from transformers import BertModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.models import Model
from settings import cnn_path, model_path, vocab_path

text = "칠개의 자연수가 주어질 때, 이들 중 홀수인 자연수들을 모두 골라 그 합을 구하고, 고른 홀수들 중 최솟값을 찾는 프로그램을 작성하시오.예를 들어, 칠개의 자연수 일십이, 칠십칠, 삼십팔, 사십일, 오십삼, 구십이, 팔십오가 주어지면 이들 중 홀수는 칠십칠, 사십일, 오십삼, 팔십오이므로 그 합은칠십칠 + 사십일 + 오십삼 + 팔십오 = 이백오십육이 되고,사십일 < 오십삼 < 칠십칠 < 팔십오이므로 홀수들 중 최솟값은 사십일이 된다."
# model_path = '/content/drive/MyDrive/kobert_from_pretrained'
# cnn_path = '/content/drive/MyDrive/model_save/cnn_weight.h5'
# vocab_path = '/content/drive/MyDrive/kobert_news_wiki_ko_cased-1087f8699e.spiece'
# cnn_state = '/content/drive/MyDrive/model_save/cnn_dict.pt'

tag_name = ['Mathematics', 'Dynamic_programming', 'Implementation', 'Graph_theory',
            'Data_structures', 'Greedy', 'String', 'Graph_traversal',
            'Bruteforcing', 'Tree', 'Binary_search', 'Number_theory',
            'Breadth_first_search', 'Depth_first_search', 'Dijkstras',
            'Divide_and_conquer', 'Stack', 'Priority_queue']

thresholds = [0.4, 0.34, 0.3, 0.35,
              0.25, 0.16, 0.28, 0.27,
              0.14, 0.18, 0.14, 0.28,
              0.19, 0.13, 0.2,
              0.1, 0.05, 0.06]


class BERToutput(nn.Module):
    def __init__(self,
                 bert):
        super(BERToutput, self).__init__()
        self.bert = bert

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        token_ids = torch.from_numpy(token_ids.reshape(1, -1))
        valid_length = torch.from_numpy(valid_length.reshape(1, -1))
        segment_ids = torch.from_numpy(segment_ids.reshape(1, -1))

        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        op, _ = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float(),
                          return_dict=False)
        x = op[:, :16, :]
        y = op[:, -48:, :]
        output = torch.cat((x, y), dim=1)

        return output


def get_model(x=0.1, num_classes=18):
    kernel_num = 16
    windows = [4, 8, 16]

    vector_input = Input(shape=(64, 768), name="vector")
    layer_conv1 = Conv1D(kernel_num, windows[0], strides=2, activation='relu')
    layer_conv2 = Conv1D(kernel_num, windows[1], strides=2, activation='relu')
    layer_conv3 = Conv1D(kernel_num, windows[2], strides=2, activation='relu')
    max_pool1 = MaxPool1D(2, 1)
    max_pool2 = MaxPool1D(2, 1)
    max_pool3 = MaxPool1D(2, 1)
    dropout = Dropout(x)

    hidden1 = layer_conv1(vector_input)
    hidden1 = max_pool1(hidden1)
    hidden1 = dropout(hidden1)

    hidden2 = layer_conv2(vector_input)
    hidden2 = max_pool2(hidden2)
    hidden2 = dropout(hidden2)

    hidden3 = layer_conv3(vector_input)
    hidden3 = max_pool3(hidden3)
    hidden3 = dropout(hidden3)

    concat = concatenate([hidden1, hidden2, hidden3], axis=1)
    out = Flatten()(concat)
    output = Dense(num_classes, name="Output", activation='sigmoid')(out)

    model = Model(inputs=vector_input, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_kobert_classifier(model_path, vocab_path, cnn_path):
    bert = BertModel.from_pretrained(model_path)
    bertmodel = BERToutput(bert)
    bertmodel.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path, padding_token='[PAD]')
    tok = nlp.data.BERTSPTokenizer(vocab_path, vocab, lower=False)
    classifier = get_model()
    classifier.load_weights(cnn_path)

    return bertmodel, tok, classifier


def convert_to_vector(bertmodel, tok, text):
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=512, pad=True, pair=False)
    sentences = transform([text])
    token_ids, valid_length, segment_ids = sentences
    vector = bertmodel(token_ids, valid_length, segment_ids)
    return vector


def classification(classifier, vector, thresholds):
    vector = vector.detach().numpy()
    #  vector = np.expand_dims(vector)

    out = classifier(vector)
    out = out.numpy().squeeze()

    ratio = [(output*100, tag) for output, thres, tag in zip(out, thresholds, tag_name)]
    ratio_dict = {}

    for one_ratio in ratio :
        ratio_dict[one_ratio[1]] = one_ratio[0]

    sdict = sorted(ratio_dict.items(), key=lambda x:x[1], reverse=True)

    result = {}

    for one_tuple in sdict[:3] :
        result[one_tuple[0]] = str(round(one_tuple[1],2)) + "%"

    label = [tag_name[i] for i in range(len(out)) if out[i] >= thresholds[i]]
    print(out)
    print(tag_name)
    print(thresholds)
    print(label)
    print(result)
    return result

bert, tokenizer, classifier = get_kobert_classifier(model_path, vocab_path, cnn_path)



