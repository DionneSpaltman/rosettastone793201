# Rosetta Stone 2

**Course:** Machine Learning 

**University:** Luiss Guido Carli

**Team Members:**  
- Daniele Biggi (793201)
- Ezgim Burcak Akseki (803311)
- Dionne Spaltman (Q00149)

Link to the sharepoint with the csv files: https://luiss-my.sharepoint.com/:f:/g/personal/dionne_spaltman_studenti_luiss_it/EiQz9X4xtYFCiIIqqVgeKJQBlsFhFWYNNqEzw2SiumfFHw?e=4BFULf 

---

## 1. Introduction

With this project, the aim was to build a model that is capable of determining how closely related two sentences are. This is also called a semantic similarity model. To do so, we used a dataset of one million sentence pairs. These served as the basis of our project. The first step in creating our model was to process these multilingual sentences. With the preprocessing we went from the input sentences to base sentences. The second step was to augment the dataset.  The third step was to build our models: we built one model based on a transformer architecture and one based on a Recurrent Neural Network architecture. Finally, we evaluated our model. 


### 1.1 Dataset Overview
The dataset that was provided, consisted of 5 columns (‘sentence1’, ‘sentence2’, ‘score’, ‘lang1’, lang2’) and 949,080 rows. An example of a sentence is ‘Ein Flugzeug hebt gerade ab.’ and its translation would be ‘An air plane is taking off.’ Each pair of sentences is assigned a score, varying from 0 to 5. 

### 1.2 Data Preprocessing
To be able to use our dataset for sentence similarity, we want to get the base form for each of the sentences. To do this, the following steps need to be performed: the text needs to be lowercase; punctuation and special characters need to be removed; sentences have to be tokenized; then we apply lemmatization; and finally stop words need to be removed (but only for languages where this makes sense).

More specifically, we starte