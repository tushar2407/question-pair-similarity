import scipy
import pickle
from question_similarity.settings import BASE_DIR
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import itertools
import re
from keras.preprocessing.sequence import pad_sequences
import keras.layers as lyr
from keras.models import Model
import numpy as np

def get_tfidf(q1, q2, ngram=(1,1)):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram, max_features=5000, token_pattern=r'\w{1,}')
    q1ngram_trans = tfidf.transform([q1, ])
    q2ngram_trans = tfidf.transform([q2, ])
    return scipy.sparse.hstack((q1ngram_trans, q2ngram_trans))

def bow_xg(q1, q2):
    CV = CountVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
    q1_trans = CV.transform([q1,])
    q2_trans = CV.transform([q2,])
    X = scipy.sparse.hstack((q1_trans, q2_trans))
    classifier1 = pickle.load(open(f"{BASE_DIR}models/analysis1/classfier1.dat", "rb"))
    return classifier1.predict(X)

def uni_xg(q1, q2): 
    X = get_tfidf(q1, q2)
    classifier2 =pickle.load(open(f"{BASE_DIR}models/analysis1/classfier2.dat", "rb"))
    return classifier2.predict(X)

def bi_xg(q1, q2):
    X = get_tfidf(q1, q2, (2,2))
    classifier3 = pickle.load(open(f"{BASE_DIR}models/analysis1/classfier3.dat", "rb"))
    return classifier3.predict(X)

def tri_xg(q1, q2):
    X = get_tfidf(q1, q2, (3,3))
    classifier4 = pickle.load(open(f"{BASE_DIR}models/analysis1/classfier4.dat", "rb"))
    return classifier4.predict(X)

def uni_lr(q1, q2):
    X = get_tfidf(q1, q2)
    clf = pickle.load(open("./models/analysis3/lr_unigram_unprocessed.pkl", "rb"))
    return clf.predict(X)

def bi_lr(q1, q2):
    X = get_tfidf(q1, q2, (2,2))
    clf = pickle.load(open("./models/analysis3/lr_bigram_unprocessed.pkl", "rb"))
    return clf.predict(X)

def tri_lr(q1, q2):
    X = get_tfidf(q1, q2, (3,3))
    clf = pickle.load(open("./models/analysis3/lr_trigram_unprocessed.pkl", "rb"))
    return clf.predict(X)
    

## check 
def create_padded_seqs(texts, counts_vectorizer, words_tokenizer, max_len=10):
    seqs = texts.apply(
        lambda s: 
            [
                counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
                for w in words_tokenizer.findall(s.lower())
            ]
        )
    return pad_sequences(seqs, maxlen=max_len)

def lstm_mlp(q1, q2):
    counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
        itertools.chain(
            [q1, ], 
            [q2, ]
        )
    )
    words_tokenizer = re.compile(counts_vectorizer.token_pattern)

    input1_tensor = lyr.Input(X1_train.shape[1:])
    input2_tensor = lyr.Input(X2_train.shape[1:])

    words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
    seq_embedding_layer = lyr.LSTM(256, activation='tanh')

    seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

    merge_layer = lyr.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

    dense1_layer = lyr.Dense(16, activation='sigmoid')(merge_layer)
    ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

    model = Model([input1_tensor, input2_tensor], ouput_layer)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    model.load_weights(f"{BASE_DIR}models/analysis2/model.pkl")

    y_pred = model.predict([q1, q2], 128)
    
    return np.where(y_pred>0.5, 1, 0)
