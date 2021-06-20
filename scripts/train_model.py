from transformers
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import pandas as pd
import datasets
import random
from IPython.display import display, HTML
import json
from functools import partial

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

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main(args):
    df = pd.read_csv(args.input_file_name, encoding='utf-8')[['zh','th']]
    
    j = {'translation':[]}
    for i in df.itertuples():
        j['translation'] += [{'zh_cn':i[1], 'th':i[2]}]
    
    train_dataset = Dataset.from_dict(j)
    raw_datasets = train_dataset.train_test_split(test_size=, seed=42)

    
    tokenizer = MarianTokenizer.from_pretrained('marian-mt-zh_cn-th')

    tokenized_datasets = raw_datasets.map(partial(preprocess_function, tokenizer=tokenizer, max_input_length = args.max_input_length ,max_target_length = args.max_target_length ,source_lang = args.source_lang , target_lang = args.target_lang), batched=True,)

    config = MarianConfig.from_pretrained('marian-mt-zh_cn-th')
    model = MarianMTModel(config)

    Training_args = Seq2SeqTrainingArguments(
        args.Train_output_dir,
        evaluation_strategy = "epoch",
        learning_rate= args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        Training_args,
        train_dataset=tokenized_datasets["train"], 
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(args.Train_output_dir)

    model.cpu()
    src_text = [
        '我爱你',
        '国王有很多心事。我明白'
    ]
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    [tokenizer.decode(t, skip_special_tokens=True) for t in translated]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--Train_output_dir', type=str)
    parser.add_argument('--source_lang', type=str)
    parser.add_argument('--target_lang', type=str)

    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--valid_size', type=float, default=0.01)


    args = parser.parse_args()
    main(args)