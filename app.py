import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import streamlit as st

st.title("Lalita Machine Translation Chinese-Thai")



option = st.sidebar.radio('', ['Chinese to Thai', 'Thai to Chinese'])

if option == 'Chinese to Thai':
    tokenizer = AutoTokenizer.from_pretrained("Lalita/marianmt-zh_cn-th")
    model = AutoModelForSeq2SeqLM.from_pretrained("Lalita/marianmt-zh_cn-th")
    src_text = st.sidebar.text_input('Enter Chinese Sentence')
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    st.success([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    

elif option == 'Thai to Chinese':
    tokenizer = AutoTokenizer.from_pretrained("Lalita/marianmt-th-zh_cn")
    model = AutoModelForSeq2SeqLM.from_pretrained("Lalita/marianmt-th-zh_cn")
    src_text = st.sidebar.text_input('Enter Thai Sentence')
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    st.success([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    