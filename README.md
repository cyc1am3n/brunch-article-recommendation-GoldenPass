## Brunch Article Recommendation - Team: GoldenPass

GoldenPass 팀의 카카오 아레나 "브런치 사용자를 위한 글 추천 대회"의 최종 제출 코드입니다.

### Dependencies

- tensorflow
- numpy
- pandas
- tqdm

---

### 사용법

#### 0. 데이터셋 경로 설정

데이터셋이 `./res` 이외에 위치한 경우는 `config.py` 를 수정하세요.

```bash
$ tree -d
// 실행 결과
```

#### 1. 필요한 라이브러리 설치

Dependencies에 적혀있는 라이브러리가 이미 설치되어 있다면 건너뛰어도 됩니다.

```bash
$ pip3 install -r requriements.txt
```

#### 2. 추천에 사용할 데이터 만들기

```bash
$ python database.py
```

`database.py` 를 실행시키면 `config.py` 에서 설정한 `data_root` 폴더에 아래와 같은 파일들이 생성됩니다.

- `article_by_testuser.json`: test 유저가 읽었던 모든 글이 dictionary로 저장된 json 파일입니다.
- `rasby_testuser_100.csv`: (추가)
- `FEB_top_58647.csv`: 2월 14일부터 3월 1일 까지 유저들이 많이 읽은 article 정보입니다.
- `MAR_article_1.csv`: 3월 1일부터 3월 7일까지 등록된 모든 article 정보입니다.
- `MAR_article_2.csv`: 3월 7일부터 3월 14일까지 등록된 모든 article 정보입니다.

#### 3. item2vec으로 article간의 유사도 구하기

```bash
$ python item2vec.py
```

`item2vec.py` 를 실행키면`config.py` 에서 설정한 `data_root` 폴더에 아래와 같은 파일들이 생성됩니다.

- `recent_article_by_user.json` : 2월 7일 이후에 읽은 글의 정보가 유저 별로 저장된 json 파일입니다.
- `word_to_id.json`: (추가)
- `id_to_word.json`: (추가)
- `word_to_id_recent.json`: (추가)
- `id_to_word_recent.json`: (추가)
- `article_embedding_matrix_recent_t.npy`: (추가)
- `article_similarity_recent_t.npy`: (추가)

#### 4. Inference

```bash
$ python inference.py
```

`inference.py` 를 실행시키면 test user가 읽을 100개의 글이 `recommend.txt` 로 저장되어 나옵니다.