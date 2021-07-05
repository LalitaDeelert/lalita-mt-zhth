This badge points to the latest released version of your repository. If you want a DOI badge for a specific release, please follow the DOI link for one of the specific releases and grab badge from the archived record.

Markdown
```
[![DOI](https://zenodo.org/badge/364486499.svg)](https://zenodo.org/badge/latestdoi/364486499)
```
reStructedText
```
.. image:: https://zenodo.org/badge/364486499.svg
   :target: https://zenodo.org/badge/latestdoi/364486499
```
HTML
```
<a href="https://zenodo.org/badge/latestdoi/364486499"><img src="https://zenodo.org/badge/364486499.svg" alt="DOI"></a>
```
Image URL
```
https://zenodo.org/badge/364486499.svg
```
Target URL
```
https://zenodo.org/badge/latestdoi/364486499
```

# marianmt

## Usage


### Train tokenizer with shared dictionary

```
python train_shared_tokenizer.py --input_fname ../data/v1/Train.csv \
	--output_dir ../models/marianmt-zh_cn-th
```

### Train model

```
export WANDB_PROJECT=marianmt-zh_cn-th
python train_model.py --input_fname ../data/v1/Train.csv \
	--output_dir ../models/marianmt-zh_cn-th \
	--source_lang zh --target_lang th \
	--metric_tokenize th_syllable --fp16
```

```
export WANDB_PROJECT=marianmt-th-zh_cn
python train_model.py --input_fname ../data/v1/Train.csv \
	--output_dir ../models/marianmt-th-zh_cn \
	--source_lang th --target_lang zh \
	--metric_tokenize zh --fp16
```
