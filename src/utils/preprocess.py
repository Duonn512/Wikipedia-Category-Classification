import torch
import re
import string

from underthesea import word_tokenize

def remove_stopwords_vietnamese(text):
    """
    Remove stopwords from the given Vietnamese text.

    Args:
        text (str): The input text to remove stopwords from.

    Returns:
        str: The filtered text with stopwords removed.
    """
    tokens = word_tokenize(text)
    with open('../data/vietnamese_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f.readlines()])
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_footnotes(text):
    """
    Removes footnotes from the given text.

    Parameters:
    text (str): The text from which footnotes need to be removed.

    Returns:
    str: The text with footnotes removed.
    """
    return re.sub(r'\[\d+\]', '', text)

def remove_punctuation(text):
    """
    Removes punctuation characters from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text with punctuation characters removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def lowercase_text(text):
    """
    Converts the given text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The text converted to lowercase.
    """
    return text.lower()

def tokenize(text):
    """
    Tokenizes the given text into words.

    Args:
        text (str): The input text.

    Returns:
        list: The list of words from the input text.
    """
    return word_tokenize(text)

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

