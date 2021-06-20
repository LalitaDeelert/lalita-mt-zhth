# marianmt

## Usage

Train tokenizer with shared dictionary

```
python train_shared_tokenizer.py --input_fname ../data/v1/Train.csv --output_dir ../models/marianmt-zh_cn-th
```

Train model

```
export WANDB_PROJECT=mariantmt-zh_cn-th
python train_model.py --input_fname ../data/v1/Train.csv \
    --output_dir ../models/marianmt-zh_cn-th --source_lang zh --target_lang th \
    --metric_tokenize th_syllable --fp16
```
