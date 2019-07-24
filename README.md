## Brunch Article Recommendation - Team: GoldenPass

GoldenPass 팀의 카카오 아레나 "브런치 사용자를 위한 글 추천 대회"의 최종 제출 코드입니다.

### Dependencies

- tensorflow
- numpy
- pandas
- tqdm

---

## 사용법

### 0. 데이터셋 경로 설정

데이터셋이 `./res/` 이외에 위치한 경우는 `config.py` 를 수정하세요.

### 1. 필요한 라이브러리 설치

Dependencies에 적혀있는 라이브러리가 이미 설치되어 있다면 건너뛰어도 됩니다.

```bash
$ pip3 install -r requirements.txt
```

### 2. 모델 만들기

```bash
$ python train.py
```

`train.py` 를 실행시키면 아래 2-1, 2-2 부분의 `database.py` 와 `item2vec.py` 가 실행됩니다.  

따라서 위 코드를 실행시키면 따로  `database.py` 와 `item2vec.py` 를 실행시킬 필요는 없습니다.

#### 2-1. 추천에 사용할 데이터 만들기

```bash
$ python database.py
```

`database.py` 를 실행시키면 `config.py` 에서 설정한 `data_root` 폴더에 아래와 같은 파일들이 생성됩니다.

- `article_by_testuser.json`: test 유저가 읽었던 모든 글이 dictionary로 저장된 json 파일입니다.
- `rasby_testuser_100.csv`: test 유저가 2월 7일 이후에 읽은 글을 최대 100개까지 저장한 csv파일 입니다.
- `FEB_top_58647.csv`: 2월 14일부터 3월 1일 까지 유저들이 많이 읽은 article 정보입니다.
- `MAR_article_1.csv`: 3월 1일부터 3월 7일까지 등록된 모든 article 정보입니다.
- `MAR_article_2.csv`: 3월 7일부터 3월 14일까지 등록된 모든 article 정보입니다.

#### 2-2. item2vec으로 article간의 유사도 구하기

```bash
$ python item2vec.py
```

`item2vec.py` 를 실행키면`config.py` 에서 설정한 `data_root` 폴더에 아래와 같은 파일들이 생성됩니다.

- `recent_article_by_user.json` : 2월 7일 이후에 읽은 글의 `article_id`가 유저 별로 저장된 json 파일입니다.
- `word_to_id_recent.json`:  `recent_article_by_user.json` 에 저장된 `article_id` 를 숫자로 토큰화 시킨 json 파일입니다.
- `id_to_word_recent.json`: 숫자로 토큰화된 데이터를 원래의 `article_id` 에 대응시킨 json 파일입니다.
- `article_embedding_matrix_recent_t.npy`: `recent_article_by_user.json` 에 저장된 정보를 바탕으로 item2vec 알고리즘을 적용해서 각 `article` 을 벡터로 임베딩시킨 .npy 파일입니다.
- `article_similarity_recent_t.npy`: 임베딩된 각 `article` 의 벡터간 유사도가 계산된 .npy 파일입니다.

### 3. Inference

```bash
$ python inference.py
```

`inference.py` 를 실행시키면 test user가 읽을 100개의 글이`{데이터셋 폴더}/{hyp}/` (default='0_SUBMIT_FINAL_TEST') 폴더에 `recommend.txt` 로 저장되어 나옵니다.

### 4. 모델 설명

`추천 알고리즘`은 다음과 같습니다.

- `1`. 최근에 읽음 && 구독자 , 최근에 읽은 빈도만큼 아래 3가지 dataset에서 추천합니다.
  - `FEB_top_30419` : 2월 14일부터 3월 1일 까지 유저들이 많이 읽은 article 정보로, read_cnt 내림차순으로 정렬되어 있음
  - `MAR_article_1`: 3월 1일부터 3월 7일까지 등록된 모든 article 정보입니다.
  - `MAR_article_2`: 3월 7일부터 3월 14일까지 등록된 모든 article 정보입니다.
  * 3가지 기간에 추천된 글들을 따로 모으고 최종적으로 한 리스트로 합쳐 추천
  - 최근에 읽은 작가 중에 구독자인 사람들의 글들을 3가지 기간에서 가져와 추천하는 방식입니다.
  - 구독자 글의 소비가 평균 35%, 글을 작성한 경과일이 지날 수록 읽은 사람이 급격히 감소한다는 것을 반영해 우선적으로 체크했습니다.
- `2`. 최근에 읽음 && 구독자의 글을 읽은 빈도가 작을 경우
  - 위와 같은 방법을 추천 글이 30개가 쌓일 때까지 반복
  - 구독자의 글들이 다른 기간에 많이 작성되었을 수도 있기때문
- `3`. 최근에 읽지 않았지만 && 구독자인 경우
  
  - 1번과 2번에서 고려된 구독자를 제외하고, 최근에 읽은 구독자의 글들도 위와 같은 방법을 통해 추가 합니다.
- `4`. 최근에 읽은 작가 && 비 구독자
  
  - 최근 경향도 반영해 주기 위해 위와 유사한 방식으로 추천합니다.
- `5`. item2vec
  
  - 남은 유저들은 최근의 읽은 글들을 역순으로 정렬해 전체 유저의 2월 7일부터의 읽은 글들의 패턴이 학습된 item2vec을 활용해 가장 비슷할 것 같은 상위 10개의 글들을 추천합니다.
- `6`. 2월 14일부터 3월 1일까지 읽은 글들의 BEST 글 추천
  
  - 위의 사항에 모두 걸리지 않았거나 남은 글들은 2월 14일부터 3월 1일까지의 글들을 가장 많이 읽은 글들부터 차례대로 추천합니다.

