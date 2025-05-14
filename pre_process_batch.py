import spacy
import string
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Load language models only once
nlp_dict = {
    "de": spacy.load("de_core_news_sm"),
    "en": spacy.load("en_core_web_sm"),
    "es": spacy.load("es_core_news_sm"),
    "fr": spacy.load("fr_core_news_sm"),
    "it": spacy.load("it_core_news_sm"),
    "ja": spacy.load("ja_core_news_sm"),
    "nl": spacy.load("nl_core_news_sm"),
    "pl": spacy.load("pl_core_news_sm"),
    "pt": spacy.load("pt_core_news_sm"),
    "ru": spacy.load("ru_core_news_sm"),
    "zh": spacy.load("zh_core_web_sm"),
}

# Define custom English stopword list
default_en_stops = nlp_dict["en"].Defaults.stop_words
not_stop = {"take", 'off', 'nor', 'no', 'through', 'elsewhere', 'anyway', 'until', 'without', 'noone', 'otherwise', 'not', 'none', 'else', 'nobody', 'anyhow', 'less', 'whatever', 'never', 'few', 'rather', 'however', 'nowhere'}

my_stop = {word for word in default_en_stops if word not in not_stop}

# Stopwords per language
stop_words = {
    "de": nlp_dict["de"].Defaults.stop_words,
    "en": my_stop,
    "es": nlp_dict["es"].Defaults.stop_words,
    "fr": nlp_dict["fr"].Defaults.stop_words,
    "it": nlp_dict["it"].Defaults.stop_words,
    "ja": nlp_dict["ja"].Defaults.stop_words,
    "nl": nlp_dict["nl"].Defaults.stop_words,
    "pl": nlp_dict["pl"].Defaults.stop_words,
    "pt": nlp_dict["pt"].Defaults.stop_words,
    "ru": nlp_dict["ru"].Defaults.stop_words,
    "zh": nlp_dict["zh"].Defaults.stop_words
}

punctuations = string.punctuation

def spacy_batch_tokenizer(texts, lang):
    if lang == "zh":
        return texts  # no processing
    tokenizer = nlp_dict[lang]
    result = []
    for doc in tokenizer.pipe(texts, batch_size=1024, n_process=1):  # set n_process>1 to enable multiprocessing
        tokens = [
            token.lemma_.lower().strip()
            for token in doc
            if token.lemma_.lower().strip() not in stop_words[lang]
            and token.lemma_.lower().strip() not in punctuations
            and token.lemma_.replace("-", "").replace("'", "").isalpha()
        ]
        result.append(" ".join(tokens))
    return result

# --- Apply on the DataFrame in batches ---
df = pd.read_csv("rs2.csv")
batch_size = 1000  # adjust this based on available RAM

def process_column_in_batches(df, text_col, lang_col, out_col):
    results = []
    for lang in tqdm(df[lang_col].unique(), desc=f"Processing {out_col} by language"):
        sub_df = df[df[lang_col] == lang]
        processed = spacy_batch_tokenizer(sub_df[text_col].tolist(), lang)
        results.append(pd.Series(processed, index=sub_df.index))
    df[out_col] = pd.concat(results).sort_index()  # maintain order

# Process both columns
process_column_in_batches(df, "sentence1", "lang1", "sentence1_clean")
process_column_in_batches(df, "sentence2", "lang2", "sentence2_clean")

# Save result
df.to_csv("rs2_pre_processed.csv", index=False)
