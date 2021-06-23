# code adapted from https://github.com/huggingface/notebooks/blob/master/examples/translation.ipynb

# # need special branch of sacrebleu for thai bleu
# !pip uninstall -q sacrebleu
# !pip install -q git+https://github.com/cstorm125/sacrebleu.git@add_thai_tokenizer
# # written with transformers 4.6.0

# export WANDB_PROJECT=mariantmt-zh_cn-th
# python train_model.py --input_fname ../data/v1/Train.csv \
#     --output_dir ../models/marianmt-zh_cn-th --source_lang zh --target_lang th \
#     --metric_tokenize th_syllable --fp16

# export WANDB_PROJECT=mariantmt-th-zh_cn
# python train_model.py --input_fname ../data/v1/Train.csv \
#     --output_dir ../models/marianmt-zh_cn-th --source_lang th --target_lang zh \
#     --metric_tokenize zh --fp16

import argparse
from transformers import (
    MarianTokenizer,
    MarianMTModel, 
    MarianConfig, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
)
from datasets import load_metric, Dataset
import numpy as np
import pandas as pd
import datasets
import random
import json
from functools import partial

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

metric = load_metric("sacrebleu")

def preprocess_function(examples, 
                        tokenizer, 
                        max_input_length,
                        max_target_length,
                        source_lang, 
                        target_lang,):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds, 
                    tokenizer,
                    metric,
                    metric_tokenize):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, 
                            references=decoded_labels,
                            tokenize= metric_tokenize)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main(args):
    df = pd.read_csv(args.input_fname, encoding='utf-8')[[args.source_lang,args.target_lang]]
    logging.info(f'Loaded {df.shape}')
    
    #convert to dictionary
    j = {'translation':[]}
    for i in df.itertuples():
        j['translation'] += [{args.source_lang: i[1], args.target_lang:i[2]}]
    
    train_dataset = Dataset.from_dict(j)
    raw_datasets = train_dataset.train_test_split(test_size=args.valid_pct, 
                                                  seed=args.seed)
    raw_datasets['train'] = raw_datasets['test'] #debug
    logging.info(f'Datasets created {raw_datasets}')
    
    tokenizer = MarianTokenizer.from_pretrained(args.output_dir)
    logging.info(f'Tokenizer loaded from {args.output_dir}')
    
    #tokenize datasets
    tokenized_datasets = raw_datasets.map(partial(preprocess_function, 
                                                  tokenizer=tokenizer,
                                                  max_input_length = args.max_input_length,
                                                  max_target_length = args.max_target_length,
                                                  source_lang = args.source_lang,
                                                  target_lang = args.target_lang), 
                                          batched=True,)
    logging.info(f'Tokenized datasets: {tokenized_datasets}')
    
    #filter those with too few tokens
    tokenized_datasets = tokenized_datasets.filter(lambda example: len(example['translation']['zh'])>2)
    tokenized_datasets = tokenized_datasets.filter(lambda example: len(example['translation']['th'])>2)
    logging.info(f'Tokenized datasets when filtered out less than 2 tokens per sequence: {tokenized_datasets}')

    config = MarianConfig.from_pretrained(args.output_dir)
    model = MarianMTModel(config)
    logging.info(f'Loaded model from {args.output_dir}')

    training_args = Seq2SeqTrainingArguments(
        args.output_dir,
        evaluation_strategy = "epoch",
        load_best_model_at_end=True,
        learning_rate= args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        seed=args.seed,
    )
    logging.info(f'Training congig {training_args}')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics,
                                tokenizer=tokenizer,
                                metric=metric,
                                metric_tokenize=args.metric_tokenize),
    )
    logging.info(f'Trainer created')

    trainer.train()
    
    model.save_pretrained(f"{args.output_dir}_best")
    tokenizer.save_pretrained(f"{args.output_dir}_best")
    logging.info(f'Best model saved')

    model.cpu()
    src_text = [
        '我爱你',
        '国王有很多心事。我明白'
    ]
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    parser.add_argument('--metric_tokenize', type=str)

    parser.add_argument('--max_input_length', type=int, default=160)
    parser.add_argument('--max_target_length', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--fp16', action='store_true')
    
    parser.add_argument('--valid_pct', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)