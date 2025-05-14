from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import torch

# Function to load model/tokenizer for a language pair (cached)
model_cache = {}


def get_model_and_tokenizer(src_lang, tgt_lang):
    if f"{src_lang}-{tgt_lang}" in model_cache:
        return model_cache[f"{src_lang}-{tgt_lang}"]

    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except:
        print(f"Direct model {model_name} not found — falling back to multilingual model.")
        if tgt_lang == 'en':
            model_name = 'Helsinki-NLP/opus-mt-mul-en'
        elif src_lang == 'en':
            model_name = 'Helsinki-NLP/opus-mt-en-mul'
        else:
            raise ValueError(f"No available translation model for {src_lang} to {tgt_lang}")

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

    model_cache[f"{src_lang}-{tgt_lang}"] = (tokenizer, model)
    return tokenizer, model


# Batched translation function
def translate_batch(texts, src_lang, tgt_lang, batch_size=16):
    tokenizer, model = get_model_and_tokenizer(src_lang, tgt_lang)
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        print(i, i + batch_size)
        batch_texts = texts[i:i + batch_size]
        print(batch_texts)
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_texts.extend(decoded)

    return translated_texts


# Function for batched back-translation
def back_translate_batch(sentences, src_langs, batch_size=1024):
    back_translated = []

    unique_src_langs = set(src_langs)
    for src_lang in tqdm(unique_src_langs, desc=f"Processing  by language"):
        print(f"Language :{src_lang}")
        intermediate_lang = "en"
        if src_lang == "en":
            intermediate_lang = "de"
        elif src_lang == "pl":
            intermediate_lang = "de"
        elif src_lang == "pt":
            intermediate_lang = "tl"

        indices = [i for i, lang in enumerate(src_langs) if lang == src_lang]
        src_sentences = [sentences[i] for i in indices]
        unique_src_sentences = list(set(src_sentences))
        print(f"Unique sentences :{len(unique_src_sentences)}")
        # Translate to intermediate
        unique_intermediate_sentences = translate_batch(unique_src_sentences, src_lang, intermediate_lang, batch_size)
        # Translate back to source
        unique_back_sentences = translate_batch(unique_intermediate_sentences, intermediate_lang, src_lang, batch_size)
        back_sentences = list(src_sentences)

        for ind in range(len(unique_src_sentences)):
            unique_word = unique_src_sentences[ind]
            back_sentences = [unique_back_sentences[ind] if x == unique_word else x for x in back_sentences]

        # Assign results back
        for idx, sent in zip(indices, back_sentences):
            back_translated.append((idx, sent))

    # Restore original order
    back_translated.sort()
    return [bt[1] for bt in back_translated]


# Language code map
lang_code_map = {
    "en": "en", "de": "de", "es": "es", "fr": "fr", "it": "it",
    "pt": "pt", "nl": "nl", "pl": "pl", "ru": "ru", "ja": "jap", "zh": "zh"
}

# Load preprocessed data
df = pd.read_csv("rs2_pre_processed.csv")
df.dropna(inplace=True)
# Common pivot
pivot_lang = "en"

# Batched back-translation for sentence1_clean
tqdm.pandas(desc="Preparing sentence1 back-translation")
sentences1 = df["sentence1_clean"].tolist()
langs1 = [lang_code_map[lang] for lang in df["lang1"]]

sentence1_bt = back_translate_batch(sentences1, langs1, batch_size=1)
df["sentence1_bt"] = sentence1_bt

# Batched back-translation for sentence2_clean
tqdm.pandas(desc="Preparing sentence2 back-translation")
sentences2 = df["sentence2_clean"].tolist()
langs2 = [lang_code_map[lang] for lang in df["lang2"]]

sentence2_bt = back_translate_batch(sentences2, langs2, batch_size=1)
df["sentence2_bt"] = sentence2_bt

# Save results
df.to_csv("rs2_backtranslated.csv", index=False)