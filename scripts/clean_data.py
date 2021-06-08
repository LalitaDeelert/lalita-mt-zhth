import argparse

import pkuseg
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from pythainlp.tokenize import word_tokenize


def main(args):
    # function to clean data containing sentence pairs
    print(args)

    # load csv
    csv_file_paths = glob.glob(os.path.join(args.input_fname, "*.csv"))
    df = pd.read_csv(csv_file_path, encoding="utf-8")

    # print stats of dataset before
    df["th_str"] = df.th.map(lambda x: len(x))
    df["zh_str"] = df.zh.map(lambda x: len(x))
    df["th_word"] = df.th.map(lambda x: len(word_tokenize(x)))
    df["zh_word"] = df.zh.map(lambda x: len(seg.cut(x)))

    df.th_str.describe()
    df.zh_str.describe()
    df.th_word.describe()
    df.zh_word.describe()
    df.ratio.describe()
    df.Similarity_score.describe()

    # deduplicate
    df.drop_duplicates()

    # calculate similarity score
    def get_similar_score(lang1: str, lang2: str, batch_size: int, embed):
        scores = []
        if len(lang1) % batch_size != 0:
            num_of_batch = int(len(lang1) / batch_size) + 1
        else:
            num_of_batch = int(len(lang1) / batch_size)

        for i in range(num_of_batch):
            start = i * batch_size
            end = start + batch_size
            if i <= num_of_batch:
                lang1_temp = lang1[start:end]
                lang2_temp = lang2[start:end]

                lang1_embedding = embed(lang1_temp)
                lang2_embedding = embed(lang2_temp)
                distance_matrix = tf.matmul(lang1_embedding, lang2_embedding, transpose_b=True).numpy()

                for j in range(len(distance_matrix)):
                    scores.append(distance_matrix[j][j])

        return scores

    emb = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3", tags=None, options=None)
    df["Similarity_score"] = get_similar_score(zh_lines, th_lines, 16, emb)

    # filter by similarity score
    min_similarity = args.min_similarity
    df = df[df["Similarity_score"] >= min_similarity]

    # calculate word ratio
    ratio = df["th_word"] / df["zh_word"]
    df["Ratio"] = ratio

    # filter by word ratio
    min_zhth_ratio = args.min_zhth_ratio
    max_zhth_ratio = args.max_zhth_ratio
    df = df[df["Ratio"] >= min_zhth_ratio and df["Ratio"] <= max_zhth_ratio]

    # print stats of dataset after
    df["th_str"] = df.th.map(lambda x: len(x))
    df["zh_str"] = df.zh.map(lambda x: len(x))
    df["th_word"] = df.th.map(lambda x: len(word_tokenize(x)))
    df["zh_word"] = df.zh.map(lambda x: len(seg.cut(x)))

    df.th_str.describe()
    df.zh_str.describe()
    df.th_word.describe()
    df.zh_word.describe()
    df.ratio.describe()
    df.Similarity_score.describe()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fname", type=str, help="Input file containing sentence pairs to clean")
    parser.add_argument("--output_fname", type=str, help="File to save as output")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase all characters")
    parser.add_argument("--min_similarity", default=0.5, type=float, help="Minimum mUSE similarity score to retain")
    parser.add_argument("--min_zhth_ratio", default=0.7, type=float, help="Minimum ZH/TH word ratio to retain")
    parser.add_argument("--max_zhth_ratio", default=1.2, type=float, help="Maximum ZH/TH word ratio to retain")

    args = parser.parse_args()
    main(args)
