# lalita-mt-zhth
[![DOI](https://zenodo.org/badge/364486499.svg)](https://zenodo.org/badge/latestdoi/364486499)

BibTeX Export
```
@software{deelert_2021_5069235,
  author       = {Deelert and
                  Polpanumas},
  title        = {LalitaDeelert/lalita-mt-zhth: First Version},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {V1.0.0},
  doi          = {10.5281/zenodo.5069235},
  url          = {https://doi.org/10.5281/zenodo.5069235}
}
```

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
