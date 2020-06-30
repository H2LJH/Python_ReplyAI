import numpy as np
import tensorflow as tf
from konlpy.tag import Okt
from pymongo import MongoClient

client = MongoClient('127.0.0.1', 27017)
db = client['local']
collection = db.get_collection('movie')
reply_list = []
okt = Okt()
all_count = 0
pos_count = 0


def mongo_select_All():
    for one in collection.find({}, {'_id': 0, 'movieName': 1, 'content': 1, 'score': 1}):
        reply_list.append([one['movieName'], one['content'], one['score']])
    return reply_list


mongo_select_All()
all_count = len(reply_list)


def read_data(filename):
    words_data = []
    with open(filename, 'r', encoding='UTF-8') as f:
        while True:
            line = f.read()[:-1]
            if not line: break
            words_data.append(line)
    return words_data


selected_words = read_data('selecteord.txt')
model = tf.keras.models.load_model('my_model.h5')


# norm은 정규화, stem은 근어로표시하기를 나타냄
def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=False)]


# 예측할 데이터의 백터화를 진행할 메서드(임베딩)
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


def predict_pos_neg(review):
    token = tokenize(review)
    tk = term_frequency(token)
    data = np.expand_dims(np.asarray(tk).astype('float32'), axis=0)
    score = float(model.predict(data))
    global pos_count
    if score > 0.5:
        pos_count += 1
        print("[{}]는 {:2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.\n".format(review, score))
    else:
        print("[{}]는 {:2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.\n".format(review, (1 - score) * 100))


def predict():
    for one in reply_list:
        predict_pos_neg(one[1])

    aCount = len(reply_list)
    pcount = pos_count
    pos_pct = (pcount * 100) / aCount
    neg_pct = 100 - pos_pct
    print('==============================================================')
    print('({}) 댓글 {}개를 감정분석한 결과').format(reply_list[0][0], aCount)
    print('긍정적인 의견 {:.2f}% / 부정적인 의견 {:2f}%').format(pos_pct, neg_pct)
    print('==============================================================')


predict()
