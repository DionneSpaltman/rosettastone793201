# Rosetta Stone

## Questions for the Checkpoint

- For the Chinese sentences, it is not able to do lemmatization. It can’t find the root word.

## Objective

An alien civilization has discovered an ancient text file on the abandoned Earth, containing millions of sentence pairs. Their challenge is to decipher how closely related these sentence pairs are, effectively developing a semantic similarity detection model.

## Dataset Features

- Contains ~1 million lines with paired sentences.
- Some sentence pairs share the same meaning, while others differ.
- Requires feature engineering and text preprocessing.

## How to Get the Dataset

The dataset consists of textual sentence pairs, requiring preprocessing steps such as tokenization, stemming, and embedding techniques to facilitate model training. It is available for download on the LUISS Learn webpage.

## Assignment

1. **Data Processing**
2. **Data Augmentation**
   - Enhance the dataset using paraphrasing techniques to improve model robustness.
3. **Model Development**
   - Design a Neural Network for sentence similarity measurement.
   - Experiment with different architectures (e.g., transformers, LSTMs).
4. **Evaluation & Reporting**
   - Test the model on unseen data and analyze performance.
   - Prepare a comprehensive report on methodology and insights.

RNN would maybe get us better results. With a LSTM (context and conceding and preceding words). It's not by chance that many powerful engines are using LSTM. They allow middle language that encapsulates the meaning from the different languages into the final. 

They are expecting evaluation metrics. RMSE or MSE. 

Directly apply a transformer to it. 

Start with LSTM. Any problems? Ask us. Don't let it for the last problem. 