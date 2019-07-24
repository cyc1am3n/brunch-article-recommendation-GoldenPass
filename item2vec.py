import tensorflow as tf
import collections
from tqdm import tqdm
import json
import argparse
import numpy as np
import pandas as pd
import random
import math

from database import load_read_data, extract_article_by_user, make_read_raw
from utils import chainer
from config import data_root

directory = data_root


def article_processing_atc_read_cnt(read):
    read_cnt_by_user, read_raw = make_read_raw(read)

    metadata = pd.read_json(directory + 'metadata.json', lines=True)
    atc = metadata
    atc['reg_datetime'] = atc['reg_ts'].apply(lambda x : datetime.fromtimestamp(x/1000.0))
    atc.loc[atc['reg_datetime'] == atc['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
    atc['reg_dt'] = atc['reg_datetime'].dt.date
    atc['type'] = atc['magazine_id'].apply(lambda x : '개인' if x == 0.0 else '매거진')
    # 컬럼명 변경
    atc.columns = ['id', 'display_url', 'article_id', 'keyword_list', 'magazine_id', 'reg_ts', 'sub_title', 'title', 'author_id', 'reg_datetime', 'reg_dt', 'type']
    atc.head()
    atc_cnt_by_reg_dt = atc.groupby('reg_dt', as_index=False)['article_id'].count()

    atc_read_cnt = read_raw[read_raw.article_id != ''].groupby('article_id')['user_id'].count()
    atc_read_cnt = atc_read_cnt.reset_index()
    atc_read_cnt.columns = ['article_id', 'read_cnt']
    #metadata 결합
    atc_read_cnt = pd.merge(atc_read_cnt, atc, how='left', left_on='article_id', right_on='article_id')
    # metadata를 찾을 수 없는 소비 로그 제외
    atc_read_cnt_nn = atc_read_cnt[atc_read_cnt['id'].notnull()]

    return atc_read_cnt_nn

def article_list_processing():
    article_by_user = {}
    article_by_user_t = dict()
    read = load_read_data(directory)
    # ITEM 2 VEC 을 위한 유저들이 최신에 읽은 글 처리
    print('Extract history of users after 2/7 for item2vec.')
    read = read[read['dt'] >= '20190207']
    read_users = list(set(list(read['user_id'])))
    
    
    for i in tqdm(range(len(read_users)), mininterval=5):
        article_by_user[i] = extract_article_by_user(read, read_users[i])

 
    for key, value in article_by_user.items():
        key = int(key)
        s = []
        value = list(reversed(value))
        for i in range(len(value)):
            if value[i] not in s:
                s.append(value[i])
        s = list(reversed(s))
        article_by_user_t[key] = s
    return article_by_user , article_by_user_t 

def make_dict():
    read = load_read_data(directory)
    atc_read_cnt_nn = article_processing(read)
    art_read_cnt_morethan_10 = atc_read_cnt_nn[atc_read_cnt_nn['read_cnt'] > 10]
    article_vocab_list = art_read_cnt_morethan_10['article_id'].tolist()
    atc_read_cnt_lessthan_10 = atc_read_cnt_nn[atc_read_cnt_nn['read_cnt'] <= 10]
    article_vocab_list_less = atc_read_cnt_lessthan_10['article_id'].tolist()

    article_vocab_list_user_t = set()
    for values in article_by_user_t.values():
        for x in values:
            article_vocab_list_user_t.add(x)

    article_vocab_list_byuser = list(set(article_vocab_list_user_t) - set(article_vocab_list_less)) #
    article_vocab_list = list(set(article_vocab_list) - set(article_vocab_list_byuser))
    nonpopular_article_vocab_list = article_vocab_list_less + article_vocab_list

    start_time = time.time()
    word_to_id = {}
    id_to_word = {}

    for word in article_vocab_list_byuser:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

    # 공백 표시할 어휘 : eos
    word_to_id['eos'] = len(word_to_id)
    id_to_word[len(word_to_id)-1] = 'eos'
    # vocab에 없는 단어는 UNK로 표시
    word_to_id['UNK'] = len(word_to_id)
    id_to_word[len(word_to_id)-1] = 'UNK'

    print ("word_to_id for pop article : %s sec"%(time.time() - start_time))

    fix_id = len(word_to_id)
    for word in nonpopular_article_vocab_list:
        word_to_id[word] = fix_id
        id_to_word[fix_id] = 'UNK'

    with open(directory + 'word_to_id_recent.json','w') as f:
        json.dump(word_to_id,f)
    
    with open(directory + 'id_to_word_recent.json','w') as f:
        json.dump(id_to_word,f)

def make_corpus():
    

    with open(directory + 'word_to_id_recent.json') as f:
        word_to_id  = json.load(f)

    with open(directory + 'id_to_word_recent.json') as f:
        id_to_word = json.load(f)

    total_article_list_by_users = list(article_by_user_t.values())
    
    random.shuffle(total_article_list_by_users)

    start_time = time.time()
    corpus = []

    for values in total_article_list_by_users:

        if(len(values) > 3):
            for x in values:
                corpus.append(x)
        
        if(len(values) > 3):
            for i in range(6):
                corpus.append('eos')
        
    corpus_r = []
    corpus_r = [word_to_id[w] for w in corpus]        
    print ("corpus created! : %s sec"%(time.time() - start_time))
    return corpus_r, word_to_id, id_to_word

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
def generate_batch(data, batch_size, num_skips, skip_window, data_index):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1                      
    assert span > num_skips

    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)   

    for i in range(batch_size // num_skips):

        targets = list(range(span))     
        targets.pop(skip_window)        
        np.random.shuffle(targets)      

        start = i * num_skips
        batch[start:start+num_skips] = buffer[skip_window]

        for j in range(num_skips):
            labels[start+j, 0] = buffer[targets[j]]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels, data_index

def train(article_by_user, article_by_user_t):
    # Step 4: skip-gram 모델 구축
    # model 구축
    make_dict()

    np.random.seed(1)
    tf.set_random_seed(1)
    vocabulary_size =(len(id_to_word)) ################################################# 수정해주기!!
    batch_size = 128        # 일반적으로 16 <= batch_size <= 512
    embedding_size = 128    # embedding vector 크기
    skip_window = 4         # target 양쪽의 단어 갯수
    num_skips = 8          # 컨텍스트로부터 생성할 레이블 갯수

    valid_size = 5     # 유사성을 평가할 단어 집합 크기
    valid_window = 15000  # 앞쪽에 있는 분포들만 뽑기 위한 샘플
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64    # negative 샘플링 갯수

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    truncated = tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size))
    nce_weights = tf.Variable(truncated)
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # embeddings 벡터.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 배치 데이터에 대해 NCE loss 평균 계산
    nce_loss = tf.nn.nce_loss(weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=embed,
                            num_sampled=num_sampled,
                            num_classes=vocabulary_size)
    loss = tf.reduce_mean(nce_loss)

    # SGD optimizer
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 유사도를 계산하기 위한 모델. 학습 모델은 optimizer까지 구축한 걸로 종료.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # train
    data, word_to_id, id_to_word = make_corpus()
    
    start_time = time.time()
    num_steps = 500001
    data = make_corpus()
    ordered_words = list(word_to_id.keys())
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        average_loss, data_index = 0, 0
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = generate_batch(data, batch_size, num_skips, skip_window, data_index)

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # 마지막 10000번에 대한 평균 loss 표시
            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                print('Average loss at step {} : {}'.format(step, average_loss))
                average_loss = 0
            
            # 10만번째마다 valid size만큼 sim 계산
            if step % 100000 == 0:
                sim = similarity.eval()         # (16, vocab_size)

                for i in range(valid_size):
                    valid_word = ordered_words[valid_examples[i]]

                    top_k = 8
                    nearest = sim[i].argsort()[-top_k - 1:-1][::-1]
                    log_str = ', '.join([ordered_words[k] for k in nearest])
                    print('Nearest to {}: {}'.format(valid_word, log_str))

        final_embeddings = normalized_embeddings.eval()

    print(time.time() - start_time , 'sec')

    np.save(directory+'article_embedding_matrix_recent_t',final_embeddings)
    print('saved_embedding_matrix')

def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def get_similarity():
    final_embeddings = np.load(directory+'article_embedding_matrix_recent_t.npy')
    
    with open(directory + 'word_to_id_recent.json') as f:
        word_to_id  = json.load(f)

    with open(directory + 'id_to_word_recent.json') as f:
        id_to_word = json.load(f)

    wl = []
    for i in tqdm(range(len(id_to_word)),mininterval = 3):
        nearest = list(cos_matrix_multiplication(final_embeddings, final_embeddings[i]).argsort()[-11:-1][::-1])
        li = []
        for x in nearest:
            li.append(id_to_word[str(x)])
        wl.append(li)
    wl =  np.asarray(wl)
    print(wl.shape)
    np.save(directory+'article_similarity_recent_t',wl)
    print('saved article_similarity_recent_t.npy')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument()

    args = parser.parse_args()
    article_by_user, article_by_user_t  = article_list_processing()
    train(article_by_user, article_by_user_t)
    get_similarity()
