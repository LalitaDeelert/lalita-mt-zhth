import argparse
from pythainlp.tokenize import word_tokenize
import jieba

def main(args):
    #function to clean data containing sentence pairs 
    print(args)
    
    #load csv
    
    #print stats of dataset before
    
    #deduplicate
    
    #calculate similarity score
    
    #filter by similarity score
    
    #calculate word ratio
    
    #filter by word ratio
    
    #print stats of dataset after
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str, 
                       help='Input file containing sentence pairs to clean')
    parser.add_argument('--output_fname', type=str, 
                       help='File to save as output')
    parser.add_argument('--lowercase', action='store_true',
                        help='Lowercase all characters')
    parser.add_argument('--min_similarity', default=0.5,
                        type=float, help='Minimum mUSE similarity score to retain')
    parser.add_argument('--min_zhth_ratio', default=0.7,
                        type=float, help='Minimum ZH/TH word ratio to retain')
    parser.add_argument('--max_zhth_ratio', default=1.2,
                        type=float, help='Maximum ZH/TH word ratio to retain')

    args = parser.parse_args()
    main(args)
    