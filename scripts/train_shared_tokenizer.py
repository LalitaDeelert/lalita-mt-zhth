# python train_shared_tokenizer.py --input_fname ../data/v1/Train.csv --output_dir ../models/marianmt-zh_cn-th

import argparse
import json
import os
import shutil
import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer
import sentencepiece as spm

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

#train tokenizer
def train_spm_tokenizer(train_fname, 
                        vocab_special_size,
                        model_dir,
                        model_prefix='both',
                        character_coverage=0.9995,
                        max_sentencepiece_length=16,
                        add_dummy_prefix='false',
                        model_type='unigram',
                        user_defined_symbols='<pad>'):
    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.train((f'--input={train_fname} '
                                   f'--model_prefix={model_prefix} '
                                   f'--vocab_size={vocab_special_size} '
                                   f'--character_coverage={character_coverage} '
                                   f'--max_sentencepiece_length={max_sentencepiece_length} '
                                   f'--add_dummy_prefix={add_dummy_prefix} '
                                   f'--model_type={model_type} '
                                   f'--user_defined_symbols={user_defined_symbols}'))
    
    #create tokenizer folder if not present
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    shutil.move('both.model', f'{model_dir}/both.model')
    shutil.move('both.vocab', f'{model_dir}/both.vocab')

def main(args):
    df = pd.read_csv(args.input_fname)[[args.source_lang, args.target_lang]]
    logging.info(f'Loaded {df.shape}')
    
    #convert to dictionary
    j = {'translation':[]}
    for i in df.itertuples():
        j['translation'] += [{args.source_lang:i[1], args.target_lang:i[2]}]
    #load into dataset
    train_dataset = Dataset.from_dict(j)
    raw_datasets = train_dataset.train_test_split(test_size=args.valid_pct, seed=args.seed)
    logging.info(f'Split at {args.valid_pct} validation set')

    #convert only train back to df
    train_df = pd.DataFrame(raw_datasets['train']['translation'])
    #save both languages to text file
    train_df.to_csv('spm_temp.txt',header=None, index=None, sep='\n')
    logging.info(f'Saved to temporary text file spm_temp.txt for training sentencepiece')

    vocab_special_size = args.vocab_size + len(args.user_defined_symbols.split(','))
    logging.info(f'Vocab size including user defined symbols = {vocab_special_size}')
    
    #train both spm
    train_spm_tokenizer(train_fname='spm_temp.txt',
                        vocab_special_size=vocab_special_size,
                        model_dir=args.output_dir,
                        character_coverage=args.character_coverage,
                        max_sentencepiece_length=args.max_sentencepiece_length,
                        add_dummy_prefix=args.add_dummy_prefix,
                        model_type=args.model_type,
                        user_defined_symbols=args.user_defined_symbols
                       ) 
    #delete temporary text file when done
    if os.path.exists("spm_temp.txt"):
        os.remove("spm_temp.txt")
    logging.info(f'Trained sentencepiece tokenizer')
    
    
    #spm model
    shutil.copyfile(f'{args.output_dir}/both.model', f'{args.output_dir}/source.spm')
    shutil.copyfile(f'{args.output_dir}/both.model', f'{args.output_dir}/target.spm')
    logging.info('Renamed sentencepiece models to source.spm and target.spm')
    
    #spm vocab
    with open(f'{args.output_dir}/both.vocab','r') as f: vocab_lines = f.readlines()
    vocab_dict = {j.split('\t')[0]:i for i,j in enumerate(vocab_lines)}
    with open(f'{args.output_dir}/vocab.json','w') as f:
        json.dump(vocab_dict,f)
    logging.info(f'Converted vocab to json format')
    
    #save config.json
    config_json = '''{
      "_num_labels": 3,
      "activation_dropout": 0.0,
      "activation_function": "swish",
      "add_bias_logits": false,
      "add_final_layer_norm": false,
      "architectures": [
        "MarianMTModel"
      ],
      "attention_dropout": 0.0,
      "bad_words_ids": [
        [
          3
        ]
      ],
      "bos_token_id": 1,
      "classif_dropout": 0.0,
      "classifier_dropout": 0.0,
      "d_model": 512,
      "decoder_attention_heads": 8,
      "decoder_ffn_dim": 2048,
      "decoder_layerdrop": 0.0,
      "decoder_layers": 6,
      "decoder_start_token_id": 3,
      "dropout": 0.1,
      "encoder_attention_heads": 8,
      "encoder_ffn_dim": 2048,
      "encoder_layerdrop": 0.0,
      "encoder_layers": 6,
      "eos_token_id": 2,
      "forced_eos_token_id": 0,
      "gradient_checkpointing": false,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2"
      },
      "init_std": 0.02,
      "is_encoder_decoder": true,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_2": 2
      },
      "max_length": 512,
      "max_position_embeddings": 512,
      "model_type": "marian",
      "normalize_before": false,
      "normalize_embedding": false,
      "num_beams": 4,
      "num_hidden_layers": 6,
      "pad_token_id": 3,
      "scale_embedding": true,
      "static_position_embeddings": true,
      "transformers_version": "4.6.0",
      "unk_token_id": 0,
      "use_cache": true,
      "vocab_size": vocab_special_size
    }'''.replace('vocab_special_size', str(vocab_special_size))

    with open(f'{args.output_dir}/config.json','w') as f:
        f.write(config_json)
    logging.info(f'Sucessfully saved config.json')
    
    tokenizer = MarianTokenizer.from_pretrained(args.output_dir)
    logging.info(f'Sucessfully saved and loaded')

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str, 
                       help='Input file containing sentence pairs to train tokenizer for MarianMT')
    parser.add_argument('--output_dir', type=str, 
                       help='Output dir to save tokenizer files')
    parser.add_argument('--source_lang', type=str, default='zh')
    parser.add_argument('--target_lang', type=str, default='th')
    
    parser.add_argument('--vocab_size', type=int, default=32000,
                       help='Vocab size')
    parser.add_argument('--user_defined_symbols', type=str, default= '<pad>',
                       help='User defined symbols for sentencepiece')
    parser.add_argument('--model_type', type=str, default= 'unigram',
                       help='Sentencepiece model type')
    parser.add_argument('--add_dummy_prefix', type=str, default= 'false',
                       help='Add dummy prefix to sentencepiece')
    parser.add_argument('--max_sentencepiece_length', type=int, default=16,
                       help='Max sentencepiece token length')
    parser.add_argument('--character_coverage', type=int, default=0.9995,
                       help='Sentencepiece character coverage')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed to split validation set')
    parser.add_argument('--valid_pct', type=float, default=0.01,
                       help='Percentage of validation set')

    args = parser.parse_args()
    main(args)