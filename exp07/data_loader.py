import pandas as pd
import numpy as np
from tensorflow import keras
from collections import Counter
import konlpy.tag import Mecab

class DataLoader():
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_train.txt')
        self.test_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_test.txt')

    def load_data(self, num_words=10000):
        tokenizer = Mecab()
        stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
        self.train_data.drop_duplicates(subset=['document'], inplace=True)
        self.train_data = self.train_data.dropna(how='any')
        self.test_data.drop_duplicates(subset=['document'], inplace=True)
        self.test_data = self.test_data.dropna(how='any')

        X_train = []
        for sentence in self.train_data['document']:
            temp_X = tokenizer.morphs(sentence)  # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
            X_train.append(temp_X)

        X_test = []
        for sentence in self.test_data['document']:
            temp_X = tokenizer.morphs(sentence)  # 토큰화
            temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
            X_test.append(temp_X)

        words = np.concatenate(X_train).tolist()
        counter = Counter(words)
        counter = counter.most_common(num_words - 4)
        vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter]
        self.word_to_index = {word: index for index, word in enumerate(vocab)}

        def wordlist_to_indexlist(wordlist):
            return [self.word_to_index[word] if word in self.word_to_index else self.word_to_index['<UNK>'] for word in wordlist]

        X_train = list(map(wordlist_to_indexlist, X_train))
        X_test = list(map(wordlist_to_indexlist, X_test))

        return X_train, np.array(list(self.train_data['label'])), X_test, np.array(list(self.test_data['label'])), self.word_to_index

    def get_maxlen(self):
        total_data_text = list(self.X_train) + list(self.X_test)
        # 텍스트데이터 문장길이의 리스트를 생성한 후
        num_tokens = [len(tokens) for tokens in total_data_text]
        num_tokens = np.array(num_tokens)
        # 문장길이의 평균값, 최대값, 표준편차를 계산해 본다.
        print('문장길이 평균 : ', np.mean(num_tokens))
        print('문장길이 최대 : ', np.max(num_tokens))
        print('문장길이 표준편차 : ', np.std(num_tokens))

        # 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,
        max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
        self.maxlen = int(max_tokens)
        print('pad_sequences maxlen : ', self.maxlen)
        print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다. '.format(np.sum(num_tokens < max_tokens) / len(num_tokens)))
        return self.maxlen

    def set_pad(self, padding='post'):
        self.X_train = keras.preprocessing.sequence.pad_sequences(self.X_train,
                                                             value=self.word_to_index["<PAD>"],
                                                             padding='post',  # 혹은 'pre'
                                                             maxlen=self.maxlen)

        self.X_test = keras.preprocessing.sequence.pad_sequences(self.X_test,
                                                            value=self.word_to_index["<PAD>"],
                                                            padding='post',  # 혹은 'pre'
                                                            maxlen=self.maxlen)
        return self.X_train, self.X_test