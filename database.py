# 라이브러리 로드
import os
from datetime import timedelta, datetime
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from config import data_root
from utils import chainer, get_class

import argparse

directory = data_root

def load_read_data(directory):
    # read data 불러오기
    read_file_lst = glob.glob(directory + 'read/*')
    exclude_file_lst = ['read.tar']

    read_df_lst = []
    for f in read_file_lst:
        file_name = os.path.basename(f)
        if file_name in exclude_file_lst:
            print(file_name)
        else:
            df_temp = pd.read_csv(f, header=None, names=['raw'])
            df_temp['dt'] = file_name[:8]
            df_temp['hr'] = file_name[8:10]
            df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]
            df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()
            read_df_lst.append(df_temp)

    read = pd.concat(read_df_lst)
    print('read data loaded.')
    return read

def make_read_raw(read):
    read_cnt_by_user = read['article_id'].str.split(' ').map(len)
    read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                             'hr': np.repeat(read['hr'], read_cnt_by_user),
                             'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                             'article_id': chainer(read['article_id'])})
    return read_cnt_by_user, read_raw

def extract_article_by_user(read, user_id):
    user_data = read[read['user_id']==user_id].sort_values(by=['dt']).reset_index(drop=True)['article_id']
    article = []
    for i in range(user_data.shape[0]):
        article += list(user_data)[i].split()
    return article

def make_user_data(read, test=True):
    # dev/test user 불러오기
    users = []
    if test:
        print('Extract history of every test users.')
        data_dir = directory + '/predict/test.users'
        file_name = 'article_by_testuser.json'
    else:
        print('Extract history of every dev users.')
        data_dir = directory + '/predict/dev.users'
        file_name = 'article_by_devuser.json'

    with open(data_dir, 'r') as fr:
        lines = fr.readlines()
        users = lines
        del lines
    for i in range(len(users)):
        users[i] = users[i].replace('\n', '')

    article_by_user = {}
    for i in tqdm(range(len(users))):
        article_by_user[i] = extract_article_by_user(read, users[i])

    with open(directory + file_name, 'w', encoding="utf-8") as make_file:
        json.dump(article_by_user, make_file, ensure_ascii=False, indent="\t")
        
def extract_article_by_user(read, user_id):
    user_data = read[read['user_id']==user_id].sort_values(by=['dt']).reset_index(drop=True)['article_id']
    article = []
    for i in range(user_data.shape[0]):
        article += list(user_data)[i].split()
    return article

def article_processing(read):
    '''
    D_A_/article_search.ipynb 참조
    < Making List >
    - rasby_testuser_100.csv : 각 testuser 가 최신에 읽은 글의 정보 (max = 100개)
    - FEB_top_58647 : 2월 14일 ~ 3월 1일간 읽은 글 best
    - MAR_article_1 : 3월 1일 ~ 3월 7일 등록된 글
    - MAR_article_2 : 3월 7일 ~ 3월 14일 등록된 글
    * 기간은 왼쪽 처음은 닫힌 구간, 오른쪽 끝은 열린 구간
    '''
    read_cnt_by_user, read_raw = make_read_raw(read)

    metadata = pd.read_json(directory + 'metadata.json', lines=True)

    atc = metadata.copy()
    atc['reg_datetime'] = atc['reg_ts'].apply(lambda x: datetime.fromtimestamp(x / 1000.0))
    atc.loc[atc['reg_datetime'] == atc['reg_datetime'].min(), 'reg_datetime'] = datetime(2090, 12, 31)
    atc['reg_dt'] = atc['reg_datetime'].dt.date
    atc['type'] = atc['magazine_id'].apply(lambda x: '개인' if x == 0.0 else '매거진')
    # 컬럼명 변경
    atc = atc.rename(columns = {"article_id" : "id", "id" : "article_id","user_id":"author_id"})
    atc.head()
    atc_cnt_by_reg_dt = atc.groupby('reg_dt', as_index=False)['article_id'].count()
    atc_read_cnt = read_raw[read_raw.article_id != ''].groupby('article_id')['user_id'].count()
    atc_read_cnt = atc_read_cnt.reset_index()
    atc_read_cnt = atc_read_cnt.rename(columns = {"user_id" : "read_cnt"})
    # metadata 결합
    atc_read_cnt = pd.merge(atc_read_cnt, atc, how='left', left_on='article_id', right_on='article_id')
    # metadata를 찾을 수 없는 소비 로그 제외
    atc_read_cnt_nn = atc_read_cnt[atc_read_cnt['id'].notnull()]
    # 클래스 적용
    atc_read_cnt_nn['class'] = atc_read_cnt_nn['read_cnt'].map(get_class)

    off_data = pd.merge(read_raw, atc, how='inner', left_on='article_id', right_on='article_id')
    off_data = off_data.rename(columns = {"dt" : "read_dt"})
    
    off_data = off_data[['read_dt', 'user_id', 'article_id', 'title', 'sub_title', 'author_id',
                         'reg_dt', 'type', 'display_url', 'keyword_list', 'magazine_id']]
    off_data['read_dt'] = pd.to_datetime(off_data['read_dt'], format='%Y%m%d')
    off_data['reg_dt'] = pd.to_datetime(off_data['reg_dt'], format='%Y-%m-%d')
    off_data['off_day'] = (off_data['read_dt'] - off_data['reg_dt']).dt.days
    off_data = pd.merge(off_data, atc_read_cnt_nn[['article_id', 'read_cnt', 'class']], how='left',
                        left_on='article_id', right_on='article_id')

    result = pd.DataFrame()

    test_users_path = directory + '/predict/test.users'
    test_users_list = []
    with open(test_users_path, 'r') as fr:
        lines = fr.readlines()
        test_users_list = lines
        del lines
    for i in range(len(test_users_list)):
        test_users_list[i] = test_users_list[i].replace('\n', '')

    for i in tqdm(range(5000), mininterval=1):
        off_data_i = off_data[off_data['user_id'] == test_users_list[i]].sort_values(by=['read_dt']).tail(100)
        result = pd.concat([result, off_data_i])

    result.to_csv(directory + 'rasby_testuser_100.csv', encoding='utf-8')
    print('Saved rasby_testuser_100.csv.')
    
    # FEB, 2월 14일부터 3월 1일까지의 인기글
    FEB_read_raw = read_raw[read_raw['dt'] >= '20190214']
    FEB_atc_read_cnt = FEB_read_raw[FEB_read_raw.article_id != ''].groupby('article_id')['user_id'].count()
    FEB_atc_read_cnt = FEB_atc_read_cnt.reset_index()
    FEB_atc_read_cnt = FEB_atc_read_cnt.rename(columns = {"user_id" : "read_cnt"})
    FEB_atc_read_cnt = pd.merge(FEB_atc_read_cnt, atc, how='left', left_on='article_id', right_on='article_id')
    # metadata를 찾을 수 없는 소비 로그 제외
    FEB_atc_read_cnt_nn = FEB_atc_read_cnt[FEB_atc_read_cnt['id'].notnull()]
    FEB_atc_read_cnt_nn['class'] = FEB_atc_read_cnt_nn['read_cnt'].map(get_class)
    FEB_top = FEB_atc_read_cnt_nn.sort_values(by='read_cnt',ascending=False)
    FEB_top = FEB_top[FEB_top['read_cnt'] >= 5]
    FEB_top.to_csv(directory + 'FEB_top_58647.csv',encoding='utf-8')
    
    # 3월 1일 ~ 3월 7일까지 등록된 글 / 3월 7일 ~ 3월 1일까지 등록된 글
    a = 20190301
    a = pd.to_datetime(a,format='%Y%m%d')
    FEB_MAR_article1  = atc[atc['reg_datetime'] >= a]
    MAR_article1 = FEB_MAR_article1[FEB_MAR_article1['reg_datetime'] >= a]
    
    b = 20190307
    b = pd.to_datetime(b,format='%Y%m%d')
    MAR_article_1 = MAR_article1[MAR_article1['reg_datetime'] < b]
    MAR_article_2 = MAR_article1[MAR_article1['reg_datetime'] >= b]
    
    c = 20190314
    c = pd.to_datetime(c,format='%Y%m%d')
    MAR_article_2 = MAR_article_2[MAR_article_2['reg_datetime'] <= c]
    
    MAR_article_1 = MAR_article_1.sort_values(by='reg_datetime')
    MAR_article_2 = MAR_article_2.sort_values(by='reg_datetime')
    
    MAR_article_1.to_csv(directory + 'MAR_article_1.csv',encoding='utf-8')
    MAR_article_2.to_csv(directory + 'MAR_article_2.csv',encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--dev', type=bool, default=False)

    args = parser.parse_args()
    read = load_read_data(directory)
    make_user_data(read, test=args.test)
    if args.test and args.dev:
        make_user_data(read, test=False)
    article_processing(read)
