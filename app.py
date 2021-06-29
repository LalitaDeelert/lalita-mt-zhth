import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("Lalita/marianmt-zh_cn-th")
model = AutoModelForSeq2SeqLM.from_pretrained("Lalita/marianmt-zh_cn-th")

st.title("Lalita Machine Translation Chinese-Thai")

src_text = st.text_input('Enter Translate Sentence')
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

st.success([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
