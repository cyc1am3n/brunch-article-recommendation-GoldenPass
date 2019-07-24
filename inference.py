import os
import json
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from config import data_root

directory = data_root

author_lists = []
class_lists = []
magazine_lists = []
unique_author = []

hyp = '0_SUBMIT_FINAL_TEST'

try:
    if not(os.path.isdir(directory + hyp )):
        os.makedirs(os.path.join(directory + hyp))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

def int_class(x):
        if x == '5%':
            result = 5
        elif x == '10%':
            result = 10
        elif x == '25%':
            result = 25
        elif x == '50%':
            result = 50
        elif x == '75%':
            result = 75
        else:
            result = 100
        return result

if __name__ == '__main__':
    # variable
    after_author = []
    after_author2 = []
    siml = []
    after_following = []
    after_recent = []
    ulfs = []
    after_sim = []

    # hyper parameter
    hyp_author = 50
    hyp_recent_article_len = 10
    hyp_top_k = 10
    hyp_read_cnt = 10
    hyp_sim = 70
    hyp = '0_SUBMIT_FINAL_TEST'

    dev_users_path = directory + 'predict/dev.users'
    dev_users_list = []
    with open(dev_users_path, 'r') as fr:
        lines = fr.readlines()
        dev_users_list = lines
        del lines
    for i in range(len(dev_users_list)):
        dev_users_list[i] = dev_users_list[i].replace('\n', '')

    test_users_path = directory + 'predict/test.users'
    test_users_list = []
    with open(test_users_path, 'r') as fr:
        lines = fr.readlines()
        test_users_list = lines
        del lines
    for i in range(len(test_users_list)):
        test_users_list[i] = test_users_list[i].replace('\n', '')

    FEB_top_58647 = pd.read_csv(directory + 'FEB_top_58647.csv')
    FEB_top_58647 = FEB_top_58647.drop('Unnamed: 0', 1)
    MAR_article_1 = pd.read_csv(directory + 'MAR_article_1.csv')
    MAR_article_1 = MAR_article_1.drop('Unnamed: 0', 1)
    MAR_article_2 = pd.read_csv(directory + 'MAR_article_2.csv')
    MAR_article_2 = MAR_article_2.drop('Unnamed: 0', 1)
    FEB_top_30419 = FEB_top_58647[:30419]
    FEB_top_30419_list = FEB_top_30419['article_id'].tolist()
    FEB_top_58647_list = FEB_top_58647['article_id'].tolist()
    print('Processed article information is successfully loaded.')

    with open(directory + 'article_by_testuser.json') as f:
        article_seen_by_testuser = json.load(f)
    
    users = pd.read_json(directory + '/users.json', lines=True)
    
    st = time.time()
    users_following_lists = [] 
    for i in range(5000):
        uid = test_users_list[i]
        ufl = list(users[users['id'] == uid]['following_list'])
        if len(ufl) > 0:
            users_following_lists.append(ufl)
        else:
            users_following_lists.append([])
    print(time.time() - st , 'sec')
    
    with open(directory + 'rasby_testuser_100.csv',encoding='utf-8') as f:
        r1 = pd.read_csv(f)
    
    r1 = r1.drop('Unnamed: 0',1)
    r1['read_dt']  = r1['read_dt'].map(lambda x : x.replace('-',''))
    a = '20190207' # 이것도 hyp 
    r2 = r1[r1['read_dt'] >= a]
    
    # 읽은 구독자 list 도 만들기
    author_lists = []
    class_lists = []
    magazine_lists = []
    unique_author = []
    for i in tqdm(range(5000),mininterval = 1): ### final 시 수정
        recent_read_raw = r2[r2['user_id'] == test_users_list[i]]  
        recent_read_raw = recent_read_raw.sort_values(by = ['read_dt'],ascending = False)
        recent_read_raw = recent_read_raw.drop_duplicates(['article_id'])
        
        len_recent_read_raw = len(recent_read_raw)
        
        # class mean
        class_sum = 0
        magazine_sum = 0
        # 최신에 읽은 작가들 정보 모으기
        author_list = []
        author_cnt = []
        for j in range(len_recent_read_raw):
            # class mean
            rrr = recent_read_raw[j:j+1]
            class_int = int_class(list(rrr['class'])[0])
            class_sum += class_int
            # magainze mean
            magazine_type = list(rrr['type'])[0]
            if(magazine_type == '매거진'):
                magazine_sum += 1
            # author list
            author = list(rrr['author_id'])[0]
            found = 0
            for k in range(len(author_list)):
                if author_list[k] == author:
                    author_cnt[k] += 1
                    found = 1
                    break
            if(found == 0):
                author_list.append(author)
                author_cnt.append(1)
    
        author_list_dict = dict()
        for j in range(len(author_list)):
            author_list_dict[author_list[j]] = author_cnt[j]
    
        if(len_recent_read_raw != 0 ):
            class_mean = class_sum / len_recent_read_raw
            class_lists.append(class_mean)
            
            magazine_mean = magazine_sum / len_recent_read_raw
            magazine_lists.append(magazine_mean)
            
            #author_list_dict = sorted(author_list_dict.items(), key=(lambda x: x[1]), reverse = True) # 이러면 [ (), () ]형태
            author_lists.append(author_list_dict)
            
            unique_author.append(len(author_list))
        else:
            class_lists.append(5.0)
            magazine_lists.append(1.0)
            author_lists.append(dict())
            unique_author.append(0)

    article_similarity = np.load(directory+'article_similarity_recent_t.npy')
    article_similarity = article_similarity.tolist()
    with open(directory + 'word_to_id_recent.json') as f:
        word_to_id  = json.load(f)
    
    with open(directory + 'id_to_word_recent.json') as f:
        id_to_word = json.load(f)
    
    print('similarity, word_to_id, id_to_word were sucessfully loaded')

    with open(directory + hyp + '/recommend' + '.txt', 'w') as file:
        for i in tqdm(range(len(test_users_list)), mininterval=10):

            recommend_all = []
            uid = test_users_list[i]
            aid = i  # author index

            for j in range(5000):
                if uid == test_users_list[j]:
                    i = j
                    break

            # 현재 유저가 읽은 following list
            ufl = users_following_lists[i]
            if (len(ufl) > 0):
                ufl = ufl[0]

            # 현재 유저가 읽은 모든 글 ,
            r = article_seen_by_testuser[str(i)]
            # 최신에 읽은 글들이 사라지지 않게 뒤집고 중복 제거하기
            r = list(reversed(r))
            s = []

            for a in range(len(r)):
                if r[a] not in s:
                    s.append(r[a])
            r = list(reversed(s))

            # 최신에 읽은 사람들 중 구독자가 있는 것부터 먼저 띄우기
            # 최신 && 구독
            author_and_cnt = author_lists[aid]
            if hyp_author > 0:

                # 일단 recent 2주 top 58647 에서
                recommend_author = []
                cnt_author = 0
                author_consider_set = set()
                ra1 = []
                ra2 = []
                ra3 = []
                for key, value in author_and_cnt.items():
                    if (len(ra1) + len(ra2) + len(ra3) >= 100): break
                    author = key
                    if author not in ufl: continue

                    cnt = value
                    author_consider_set.add(key)
                    # 2월 14일 ~ 3월 1일간 인기글 30419개 중에 있으면 뽑기
                    author_all_article_1 = FEB_top_30419[FEB_top_30419['author_id'] == author].sort_values(by='reg_dt',
                                                                                                           ascending=False)  # 최신에 작성한 글들의 인기순이겠지 # 더 줄여봐도 될 듯
                    cnt_1 = 0
                    for j in range(len(author_all_article_1)):
                        if (cnt_1 >= cnt): break
                        artic = list(author_all_article_1[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                ra1.append(artic)
                                cnt_1 += 1

                    #  3월 1일 ~ 3월 7일에 쓰여진 글 있으면 append
                    author_all_article_2 = MAR_article_1[MAR_article_1['author_id'] == author]
                    cnt_2 = 0
                    for j in range(len(author_all_article_2)):
                        if (cnt_2 >= cnt): break
                        artic = list(author_all_article_2[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                ra2.append(artic)
                                cnt_2 += 1

                    #  3월 7일 ~ 3월 14일에 쓰여진 글 있으면 append
                    author_all_article_3 = MAR_article_2[MAR_article_2['author_id'] == author]
                    cnt_3 = 0
                    for j in range(len(author_all_article_3)):
                        if (cnt_3 >= cnt): break
                        artic = list(author_all_article_3[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                ra3.append(artic)
                                cnt_3 += 1

                # NDCG 반영
                recommend_author = ra1 + ra2 + ra3
                for artic in recommend_author:
                    if artic not in recommend_all:
                        recommend_all.append(artic)

            if (len(recommend_all) > 100):
                recommend_all = recommend_all[:100]

            after_author.append(len(recommend_all))

            # 최신 && 구독 minimum > 50 개 채우기
            if (len(recommend_all) > 0):  # 최신에 읽은 글이 있다는 소리고
                iter_1_cnt = 0
                ra1 = []
                ra2 = []
                ra3 = []
                recommend_author = []
                while (len(recommend_author) < 30):  # 최신의 글을 고려해서 최대 60~ 개 까지만 넣어주겠다 # 구독작가 && 최신 읽은 작가
                    # 혹시 읽은 작가 중에 2-14 - 3-14 까지 쓴 글이 없을 수 있으므로, 적게 뽑힐 수 있으므로 60까지는 구독자가 읽은 글들로 채우기
                    if (iter_1_cnt >= 30): break
                    iter_1_cnt += 1

                    cnt_author = 0
                    for key, value in author_and_cnt.items():
                        author = key
                        if (len(recommend_author) >= 35): break
                        if author in ufl:

                            cnt = value
                            # 2월 14일 ~ 3월 1일간 인기글 30419개 중에 있으면 뽑기
                            author_all_article_1 = FEB_top_30419[FEB_top_30419['author_id'] == author].sort_values(
                                by='reg_dt', ascending=False)  # 최신에 작성한 글들의 인기순이겠지 # 더 줄여봐도 될 듯
                            cnt_1 = 0
                            for j in range(len(author_all_article_1)):
                                if (cnt_1 >= cnt): break
                                artic = list(author_all_article_1[j:j + 1]['article_id'])[0]
                                if artic not in r:
                                    if artic not in recommend_all:
                                        ra1.append(artic)
                                        recommend_author.append(artic)
                                        cnt_1 += 1

                            #  3월 1일 ~ 3월 7일에 쓰여진 글 있으면 append
                            author_all_article_2 = MAR_article_1[MAR_article_1['author_id'] == author]
                            cnt_2 = 0
                            for j in range(len(author_all_article_2)):
                                if (cnt_2 >= cnt): break
                                artic = list(author_all_article_2[j:j + 1]['article_id'])[0]
                                if artic not in r:
                                    if artic not in recommend_all:
                                        ra2.append(artic)
                                        recommend_author.append(artic)
                                        cnt_2 += 1

                            #  3월 7일 ~ 3월 14일에 쓰여진 글 있으면 append
                            author_all_article_3 = MAR_article_2[MAR_article_2['author_id'] == author]
                            cnt_3 = 0
                            for j in range(len(author_all_article_3)):
                                if (cnt_3 >= cnt): break
                                artic = list(author_all_article_3[j:j + 1]['article_id'])[0]
                                if artic not in r:
                                    if artic not in recommend_all:
                                        ra3.append(artic)
                                        recommend_author.append(artic)
                                        cnt_3 += 1
                recommend_author = ra1 + ra2 + ra3

                for artic in recommend_author:
                    if artic not in recommend_all:
                        recommend_all.append(artic)

            if (len(recommend_all) > 100):
                recommend_all = recommend_all[:100]

            after_author2.append(len(recommend_all))

            # 최근 안 읽은 && 구독자
            ufl_not_yet = []
            for t in range(len(ufl)):
                if ufl[t] not in author_consider_set:
                    ufl_not_yet.append(ufl[t])

            ufls_i = 0
            if len(ufl_not_yet) > 0:
                for author in ufl_not_yet:  #
                    cnt_1 = 0
                    cnt_2 = 0
                    cnt_3 = 0

                    if (len(recommend_all) >= 100): break
                    author_all_article_1 = FEB_top_30419[FEB_top_30419['author_id'] == author].sort_values(by='reg_dt',
                                                                                                           ascending=False)  # top 50000 or to 10000 으로 바꿔서 비
                    # 너무 많은 글들이 있으면 10개로 자르자
                    if (len(author_all_article_1) > 10):
                        author_all_article_1 = author_all_article_1[:10]
                    for j in range(len(author_all_article_1)):
                        if (len(recommend_all) >= 100): break
                        if (cnt_1 >= 3): break
                        artic = list(author_all_article_1[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_all:
                                recommend_all.append(artic)
                                ufls_i += 1
                                cnt_1 += 1

                    #  3월 1일 ~ 3월 7일에 쓰여진 글 있으면 append
                    author_all_article_2 = MAR_article_1[MAR_article_1['author_id'] == author]
                    if (len(author_all_article_2) > 10):
                        author_all_article_2 = author_all_article_2[:10]
                    for j in range(len(author_all_article_2)):
                        if (len(recommend_all) >= 100): break
                        if (cnt_2 >= 3): break
                        artic = list(author_all_article_2[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_all:
                                recommend_all.append(artic)
                                ufls_i += 1
                                cnt_2 += 1

                    #  3월 7일 ~ 3월 14일에 쓰여진 글 있으면 append
                    author_all_article_3 = MAR_article_2[MAR_article_2['author_id'] == author]
                    if (len(author_all_article_3) > 10):
                        author_all_article_3 = author_all_article_3[:10]
                    for j in range(len(author_all_article_3)):
                        if (len(recommend_all) >= 100): break
                        if (cnt_3 >= 3): break
                        artic = list(author_all_article_3[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_all:
                                recommend_all.append(artic)
                                ufls_i += 1
                                cnt_3 += 1

            if (len(recommend_all) > 100):
                recommend_all = recommend_all[:100]

            ulfs.append(ufls_i)
            after_following.append(len(recommend_all))

            # 최신에 읽음 && 구독자 아님
            if hyp_author > 0:

                # 일단 recent 2주 top 58647 에서
                recommend_author = []
                cnt_author = 0
                author_consider_set = set()

                for key, value in author_and_cnt.items():
                    author = key
                    if author in ufl: continue

                    cnt = value
                    author_consider_set.add(key)
                    # 2월 14일 ~ 3월 1일간 인기글 30419개 중에 있으면 뽑기
                    author_all_article_1 = FEB_top_30419[FEB_top_30419['author_id'] == author].sort_values(by='reg_dt',
                                                                                                           ascending=False)  # 최신에 작성한 글들의 인기순이겠지 # 더 줄여봐도 될 듯
                    cnt_1 = 0
                    for j in range(len(author_all_article_1)):
                        if (cnt_1 >= cnt): break
                        artic = list(author_all_article_1[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                cnt_1 += 1

                    #  3월 1일 ~ 3월 7일에 쓰여진 글 있으면 append
                    author_all_article_2 = MAR_article_1[MAR_article_1['author_id'] == author]
                    cnt_2 = 0
                    for j in range(len(author_all_article_2)):
                        if (cnt_2 >= cnt): break
                        artic = list(author_all_article_2[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                cnt_2 += 1

                    #  3월 7일 ~ 3월 14일에 쓰여진 글 있으면 append
                    author_all_article_3 = MAR_article_2[MAR_article_2['author_id'] == author]
                    cnt_3 = 0
                    for j in range(len(author_all_article_3)):
                        if (cnt_3 >= cnt): break
                        artic = list(author_all_article_3[j:j + 1]['article_id'])[0]
                        if artic not in r:
                            if artic not in recommend_author:
                                recommend_author.append(artic)
                                cnt_3 += 1

                for artic in recommend_author:
                    if artic not in recommend_all:
                        recommend_all.append(artic)

            # 100 개 이상 추천될 수 있으니 자르기 | hyper parameter 만큼 자르기
            if len(recommend_all) >= 100:
                recommend_all = recommend_all[:100]

            after_recent.append(len(recommend_all))

            # 최신에 본 글 위주로 similarity 조사
            # 최신에 본 글 + similarity (전체 유저)
            if (len(r) >= hyp_read_cnt):
                wtil = list(word_to_id.keys())
                # countt +=1
                if (hyp_recent_article_len > len(r)):
                    hyp_recent_article_len = len(r)
                r = r[-hyp_recent_article_len:]
                r = list(reversed(r))
                siml_cnt = 0
                for j in range(len(r)):
                    if (siml_cnt >= hyp_sim): break
                    if (r[j] not in wtil):
                        r[j] = 'UNK'
                    s = article_similarity[word_to_id[r[j]]]
                    sim_list = s[:hyp_top_k]
                    for k in range(len(sim_list)):
                        if len(recommend_all) >= 100: break

                        if sim_list[k] not in recommend_all:
                            recommend_all.append(sim_list[k])
                            siml_cnt += 1
                            if (siml_cnt >= hyp_sim): break
                siml.append(siml_cnt)
                if len(recommend_all) >= 100:
                    recommend_all = recommend_all[:100]

            after_sim.append(len(recommend_all))

            if (len(recommend_all) >= 100):
                recommend_all = recommend_all[:100]
            else:
                # 남으면 FEB_top 중 아무거나 뽑아서 주고
                # following list 있으면 여기에 활용해도 좋을듯
                # cntttt+=1

                # 나머지는 그냥 2월 14일 - 3월 1일까지 글들 중 top 뽑기
                for x in FEB_top_30419_list:
                    if x not in recommend_all:
                        recommend_all.append(x)
                        if (len(recommend_all) == 100): break

            # 다시 한 번 더 점검
            if (len(recommend_all) > 100):
                recommend_all = recommend_all[:100]

            recommend_normal = ''
            for x in recommend_all:
                recommend_normal += ' ' + x

            file.write(uid + recommend_normal)
            file.write('\n')