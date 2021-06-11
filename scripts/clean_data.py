# python clean_data.py --input_fname ../data/TED_zhth_ALL.csv \
# --output_fname ../data/ted_zhth_cleaned.csv --batch_size 128 \
# --min_similarity 0.5 --min_zhth_ratio 0.5 --max_zhth_ratio 2.0

import argparse
import glob
from tqdm.auto import tqdm
import pandas as pd
from pythainlp.tokenize import word_tokenize
import pkuseg
seg = pkuseg.pkuseg()

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

#calculate similarity score
emb = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3', tags=None, options=None)
def get_similar_score(lang1: str, lang2: str, batch_size: int, embed):
    scores = []
    if len(lang1) % batch_size != 0:
        num_of_batch = int(len(lang1)/batch_size)+1
    else:
        num_of_batch = int(len(lang1)/batch_size)

    for i in tqdm(range(num_of_batch)):
        start = i*batch_size
        end = start+batch_size
        if i <= num_of_batch:
            lang1_temp = lang1[start:end]
            lang2_temp = lang2[start:end]

            lang1_embedding = embed(lang1_temp)
            lang2_embedding = embed(lang2_temp)
            distance_matrix = tf.matmul(
                lang1_embedding, lang2_embedding, transpose_b=True).numpy()

            for j in range(len(distance_matrix)):
                scores.append(distance_matrix[j][j])

    return scores

def main(args):
    #function to clean data containing sentence pairs 
    print(args)
    
    #load csv
    df = pd.read_csv(args.input_fname, encoding='utf-8')
    print(f'loaded {df.shape}')
    
    #deduplicate
    df = df.drop_duplicates()
    df = df.groupby('zh').th.max().reset_index()
    df = df.groupby('th').zh.max().reset_index()
    print(f'deduplicated to {df.shape}')
    
    #print stats of dataset before
    df['th_char'] = df.th.map(lambda x: len(x))
    df['zh_char'] = df.zh.map(lambda x: len(x))
    df['th_word'] = df.th.map(lambda x: len(word_tokenize(x)))
    df['zh_word'] = df.zh.map(lambda x: len(seg.cut(x)))
    print('calculated char/word statistics')
    
    #calculate similarity score
    df['similarity_score'] = get_similar_score(df.zh.tolist(), df.th.tolist(), args.batch_size, emb)
    print('calculated mUSE similarity scores')
    
    #filter by similarity score
    df = df[df['similarity_score'] >= args.min_similarity]
    print('filtered by mUSE similarity scores')
    
    #calculate word ratio
    df['zhth_ratio'] = df['zh_word'] / df['th_word']
    print('calculated zhth ratios')
    
    #filter by word ratio
    df = df[df['zhth_ratio'].map(lambda x: (x >= args.min_zhth_ratio) & (x <= args.max_zhth_ratio))]
    print('filtered by zhth word ratios')
    
    #print stats of dataset after
    print('zh/th characters')
    print(df.zh_char.describe())
    print(df.th_char.describe())
    print('zh/th words')
    print(df.zh_word.describe()) 
    print(df.th_word.describe()) 
    print('zh/th ratios')
    print(df.zhth_ratio.describe()) 
    print('zh/th similarity scores')
    print(df.similarity_score.describe())
    
    #save
    df.to_csv(args.output_fname, index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str, 
                       help='Input file containing sentence pairs to clean')
    parser.add_argument('--output_fname', type=str, 
                       help='File to save as output')
    parser.add_argument('--lowercase', action='store_true',
                        help='Lowercase all characters')
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch size to perform mUSE similarity')
    parser.add_argument('--min_similarity', default=0.5,
                        type=float, help='Minimum mUSE similarity score to retain')
    parser.add_argument('--min_zhth_ratio', default=0.5,
                        type=float, help='Minimum ZH/TH word ratio to retain')
    parser.add_argument('--max_zhth_ratio', default=2.0,
                        type=float, help='Maximum ZH/TH word ratio to retain')

    args = parser.parse_args()
    main(args)
    