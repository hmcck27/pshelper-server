import torch
from torch import nn
import gluonnlp as nlp
import numpy as np
from transformers import BertModel
# from settings import model_path, vocab_path, cnn_path

text = "칠개의 자연수가 주어질 때, 이들 중 홀수인 자연수들을 모두 골라 그 합을 구하고, 고른 홀수들 중 최솟값을 찾는 프로그램을 작성하시오.예를 들어, 칠개의 자연수 일십이, 칠십칠, 삼십팔, 사십일, 오십삼, 구십이, 팔십오가 주어지면 이들 중 홀수는 칠십칠, 사십일, 오십삼, 팔십오이므로 그 합은칠십칠 + 사십일 + 오십삼 + 팔십오 = 이백오십육이 되고,사십일 < 오십삼 < 칠십칠 < 팔십오이므로 홀수들 중 최솟값은 사십일이 된다."
model_path = '/content/drive/MyDrive/kobert_from_pretrained'
cnn_path = '/content/drive/MyDrive/model_save/freeze_cnn.pt'
vocab_path = '/content/drive/MyDrive/kobert_news_wiki_ko_cased-1087f8699e.spiece'

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


class Classifier(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_classes=8,
                 dr_rate=0.0):
        super(Classifier, self).__init__()
        # 16, 2848
        # 32, 5696
        # 1312
        self.kernel_num = 16
        self.conv1d_maxpooling1 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 4, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )
        self.conv1d_maxpooling2 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )
        self.conv1d_maxpooling3 = nn.Sequential(
            nn.Conv1d(hidden_size, self.kernel_num, 16, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Dropout(dr_rate)
        )

        self.classifier = nn.Linear(1312, num_classes)

    def forward(self, x):
        out1 = self.conv1d_maxpooling1(x.transpose(1, 2))
        out2 = self.conv1d_maxpooling2(x.transpose(1, 2))
        out3 = self.conv1d_maxpooling3(x.transpose(1, 2))
        out = torch.cat((out1, out2, out3), 2)
        out = out.reshape(out.size(0), -1)

        return self.classifier(out)


def get_kobert_classifier(model_path, vocab_path, cnn_path):
    bert = BertModel.from_pretrained(model_path)
    bertmodel = BERToutput(bert)
    bertmodel.eval()
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path, padding_token='[PAD]')
    tok = nlp.data.BERTSPTokenizer(vocab_path, vocab, lower=False)
    classifier = torch.load(cnn_path, map_location=torch.device('cpu'))
    classifier.eval()

    return bertmodel, tok, classifier


def convert_to_vector(bertmodel, tok, text):
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=512, pad=True, pair=False)
    sentences = transform([text])
    token_ids, valid_length, segment_ids = sentences
    vector = bertmodel(token_ids, valid_length, segment_ids)

    return vector


def classification(classifier, vector, thresholds):
    out = classifier(vector)
    out = torch.sigmoid(out).squeeze(0).tolist()
    label = [tag_name[i] for i in range(len(out)) if out[i] >= thresholds[i]]

    return label



if __name__ == '__main__':
    bert, tokenizer, classifier = get_kobert_classifier(model_path, vocab_path, cnn_path)
    vector = convert_to_vector(bert, tokenizer, text)
    out = classification(classifier, vector, thresholds)
    print(out)
    #  return out
