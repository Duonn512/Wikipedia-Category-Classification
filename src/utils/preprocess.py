import torch
import re
import string

from underthesea import word_tokenize

def remove_stopwords_vietnamese(text):
    tokens = word_tokenize(text)
    with open('../data/vietnamese_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f.readlines()])
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_footnotes(text):
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    return cleaned_text

def remove_punctuation(text):
    punctuation_chars = string.punctuation
    cleaned_text = ''.join([char for char in text if char not in punctuation_chars])
    return cleaned_text

def tokenize(text):
    list_word = word_tokenize(text)
    return list_word

def custom_transform(x, w2v_model, TX=80):
    """
    Define preprocess steps for DeepRNN, BiGRU, BiLSTM, BiLSTM + Attention, TextCNN
    Steps:
    - Lowercase
    - Remove footnotes
    - Remove punctuation
    - Tokenize
    - Pad or truncate to fixed length
    - Convert to word index
    """
    x = x.lower()
    x = remove_footnotes(x)
    x = remove_punctuation(x)
    x = x.split()
    if len(x) < TX:
        x = x + ['<PAD>'] * (TX - len(x))
    elif len(x) > TX:
        x = x[:TX]
    x = [w2v_model.key_to_index[word] if word in w2v_model.key_to_index else w2v_model.key_to_index['<UNK>'] for word in x]
    x = torch.tensor(x, dtype=torch.long)  # Ensure the tensor is of type long
    return x

