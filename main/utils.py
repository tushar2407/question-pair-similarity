# import scipy
import pickle
from question_similarity.settings import BASE_DIR
import re
from string import punctuation
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import numpy as np

def pad_str(s):
    return ' '+s+' '

def normalize_text(text):
    SPECIAL_TOKENS = {'non-ascii': 'non_ascii_word'}

    if pd.isnull(text) or len(text)==0:
        return ''

    text = text.lower()
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) 
    text = [word for word in text if word not in punctuation]
    text = ''.join(text)

    return text

# def get_tfidf(q1, q2, ngram="unigram"):
#     tfidf = pickle.load(open(f"{BASE_DIR}/models/analysis1/{ngram}.pkl", "rb"))
#     q1ngram_trans = tfidf.transform([q1, ])
#     q2ngram_trans = tfidf.transform([q2, ])
#     return scipy.sparse.hstack((q1ngram_trans, q2ngram_trans))

# def bow_xg(q1, q2):
#     CV = pickle.load(open(f"{BASE_DIR}/models/analysis1/bow.pkl", "rb"))
#     q1_trans = CV.transform([q1,])
#     q2_trans = CV.transform([q2,])
#     X = scipy.sparse.hstack((q1_trans, q2_trans))
#     classifier1 = pickle.load(open(f"{BASE_DIR}/models/analysis1/bow_xg.pkl", "rb"))
#     return classifier1.predict(X)

# def uni_xg(q1, q2): 
#     X = get_tfidf(q1, q2)
#     classifier2 =pickle.load(open(f"{BASE_DIR}/models/analysis1/classfier2.dat", "rb"))
#     return classifier2.predict(X)

# def bi_xg(q1, q2):
#     X = get_tfidf(q1, q2, "bigram")
#     classifier3 = pickle.load(open(f"{BASE_DIR}/models/analysis1/classfier3.dat", "rb"))
#     return classifier3.predict(X)

# def tri_xg(q1, q2):
#     X = get_tfidf(q1, q2, "trigram")
#     classifier4 = pickle.load(open(f"{BASE_DIR}/models/analysis1/classfier4.dat", "rb"))
#     return classifier4.predict(X)

# def uni_lr(q1, q2):
#     X = get_tfidf(q1, q2)
#     clf = pickle.load(open(f"{BASE_DIR}/models/analysis3/lr_unigram_unprocessed.pkl", "rb"))
#     return clf.predict(X)

# def bi_lr(q1, q2):
#     X = get_tfidf(q1, q2, "bigram")
#     clf = pickle.load(open(f"{BASE_DIR}/models/analysis3/lr_bigram_unprocessed.pkl", "rb"))
#     return clf.predict(X)

# def tri_lr(q1, q2):
#     X = get_tfidf(q1, q2, "trigram")
#     clf = pickle.load(open(f"{BASE_DIR}/models/analysis3/lr_trigram_unprocessed.pkl", "rb"))
#     return clf.predict(X)
    

# # ## check 
def create_padded_seqs(texts, max_len=10):
    counts_vectorizer = pickle.load(open(f"{BASE_DIR}/models/analysis2/bow.pkl", "rb"))
    other_index = len(counts_vectorizer.vocabulary_)
    words_tokenizer = re.compile(counts_vectorizer.token_pattern)
    seqs = list(map(lambda s: 
            [
                counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
                for w in words_tokenizer.findall(s.lower())
            ], texts))
    # texts.apply(
    #     lambda s: 
    #         [
    #             counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
    #             for w in words_tokenizer.findall(s.lower())
    #         ]
    #     )
    return pad_sequences(seqs, maxlen=max_len)

def lstm_mlp(q1, q2):
    model = load_model(f"{BASE_DIR}/models/analysis2/model.h5")
    x1 = create_padded_seqs(np.array([q1,]))
    x2 = create_padded_seqs(np.array([q2,]))
    y_pred = model.predict([x1, x2],128)
    return np.where(y_pred>0.5, 1, 0)
