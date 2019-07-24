import argparse

from database import *
from item2vec import *
from config import data_root

directory = data_root

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 매개변수 필요하면 넣기
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--dev', type=bool, default=False)
    # parser.add_argument()
    args = parser.parse_args()

    read = load_read_data(directory)
    make_user_data(read, test=args.test)
    if args.test and args.dev:
        make_user_data(read, test=False)
    article_processing(read)

    article_by_user, article_by_user_t = article_list_processing()
    train(article_by_user, article_by_user_t)
    get_similarity()