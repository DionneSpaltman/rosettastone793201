import spacy
import string


nlp_dict ={
    "de": spacy.load("de_core_news_sm"),
    "en" : spacy.load("en_core_web_sm"),
    "es": spacy.load("es_core_news_sm"),
    "fr": spacy.load("fr_core_news_sm"),
    "it": spacy.load("it_core_news_sm"),
    "ja": spacy.load("ja_core_news_sm"),
    "nl": spacy.load("nl_core_news_sm"),
    "pl": spacy.load("pl_core_news_sm"),
    "pt": spacy.load("pt_core_news_sm"),
    "ru": spacy.load("ru_core_news_sm"),
    "zh": spacy.load("zh_core_web_sm")
}

stop_words = nlp_dict["en"].Defaults.stop_words
not_stop = {"take" , 'nor', 'no', 'through', 'elsewhere', 'anyway', 'until', 'without', 'noone', 'otherwise', 'not', 'none', 'else', 'nobody', 'anyhow', 'less', 'whatever', 'never', 'few', 'rather', 'however', 'nowhere'}

# Initialize an empty set to hold the customized stop words
my_stop = set([])

# Iterate over the default stop words set
for i in stop_words:
    # If a default stop word is not in the not_stop set (i.e., it's not a word we want to keep), add it to the custom stop words set
    if i not in not_stop:
        my_stop.add(i)

stop_words = {
    "de": nlp_dict["de"].Defaults.stop_words,
    "en" : my_stop,
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



def spacy_tokenizer(sentence ,language_code):
    # Tokenize, lemmatize, filter stopwords/punctuations, and check if alphabetical in one step
    if language_code == "zh":
        return sentence
    tokens = [
        word.lemma_.lower().strip()
        for word in nlp_dict[language_code](sentence)
        if word.lemma_.lower().strip() not in stop_words[language_code]
        and word.lemma_.lower().strip() not in punctuations
        and word.lemma_.isalpha() # alphabetic character check
    ]
    return " ".join(tokens)

# sentence = "An air plane is taking off."
# print(spacy_tokenizer(sentence,"en"))

test_sentences = {
    "en": "The airplane is taking off.",
    "de": "Ein Flugzeug hebt gerade ab.",
    "es": "Un avión está despegando.",
    "fr": "Un avion est en train de décoller.",
    "it": "Un aereo sta decollando.",
    "pt": "Um avião está decolando.",
    "nl": "Een vliegtuig is aan het opstijgen.",
    "pl": "Samolot właśnie startuje.",
    "ru": "Самолет взлетает.",
    "ja": "飛行機が離陸します。",
    "zh": "飞机正在起飞。"
}

for lang, sentence in test_sentences.items():
    print(f"{lang.upper()} | Original: {sentence}")
    print(f"         Preprocessed: {spacy_tokenizer(sentence, lang)}\n")

from tqdm import tqdm
import pandas as pd

df = pd.read_csv("rs2.csv")

tqdm.pandas(desc="Preprocessing sentence1")
df["sentence1_clean"] = df.progress_apply(
    lambda row: spacy_tokenizer(row["sentence1"], row["lang1"]), axis=1
)

tqdm.pandas(desc="Preprocessing sentence2")
df["sentence2_clean"] = df.progress_apply(
    lambda row: spacy_tokenizer(row["sentence2"], row["lang2"]), axis=1
)
df.to_csv("rs2_pre_processed.csv", index=False)

