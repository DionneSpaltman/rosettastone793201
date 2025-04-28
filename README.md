# Rosetta Stone 2

### Objective
An alien civilization has discovered an ancient text file on the abandoned Earth, containing
millions of sentence pairs. Their challenge is to decipher how closely related these sentence
pairs are, effectively developing a semantic similarity detection model.

### Dataset Features
- Contains ~1 million lines with paired sentences.
- Some sentence pairs share the same meaning, while others differ.
- Requires feature engineering and text preprocessing.

### How to Get the Dataset
The dataset consists of textual sentence pairs, requiring preprocessing steps such as
tokenization, stemming, and embedding techniques to facilitate model training. It is
available for download on LUISS learn webpage.

### Assignment
1. **Data Processing**  
   - Apply NLP techniques such as word embeddings, stemming, lemmatization, and stop-word removal.
- You clean and prepare the raw text data so the model can understand it better.
- Why? Raw text is messy — models need clean numerical inputs.

2. **Data Augmentation**  
   - Enhance the dataset using paraphrasing techniques to improve model robustness.
- Create new, slightly different versions of your existing sentences.
- Why? It makes your model see more variations and generalize better to unseen sentences.

3. **Model Development**  
   - Design a Neural Network for sentence similarity measurement.  
   - Experiment with different architectures (e.g., transformers, LSTMs).
- Build a deep learning model that can compare two sentences and predict how similar they are.

4. **Evaluation & Reporting**  
   - Test the model on unseen data and analyze performance.  
   - Prepare a comprehensive report on methodology and insights.
- Explain what you did at each step: How you cleaned data, how you augmented it, What model you built (with hyperparameters), Results and what you learned, Include graphs if you can (loss curves, confusion matrix, etc.).

### Deadlines
The following deadlines apply for the project:
- DONE March 13th: Submit your groups and preferences 
- DONE Second week of April: Mid check of the project – via TAs Webex (luiss.webex.com/meet/fangeletti, luiss.webex.com/meet/ampanti)
- May 9th: Project pitch (final) – via TAs Webex (luiss.webex.com/meet/fangeletti, luiss.webex.com/meet/ampanti)
- May 12th, 23:59: Final submission of the repository (Each group must submit via mail to
gitaliano@luiss.it, fangeletti@luiss.it and ampanti@luiss.it, the URL of a GitHub repository)

Kindly ensure that you meet these deadlines, as they are essential for the successful
completion of the project.

RNN would maybe get us better results. With a LSTM (context and conceding and preceding words). It's not by chance that many powerful engines are using LSTM. They allow middle language that encapsulates the meaning from the different languages into the final. 

They are expecting evaluation metrics. RMSE or MSE. 

Directly apply a transformer to it. 

Start with LSTM. Any problems? Ask us. Don't let it for the last problem. 