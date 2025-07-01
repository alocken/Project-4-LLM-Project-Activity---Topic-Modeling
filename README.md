# LLM Project

## Overall Project Task
Develop a Topic Modeling tool (NLP project using a pre-trained language model), that includes: 
- selecting a topic and dataset,
- setting up project workspace,
- applying text processing and tokenization techniques,
- utilizing transfer learning for model development
- evaluating and optimizing model performance, and
- completed final project report

Theme will look to identify underlying topics/themes from the large collections of news groups.

## Dataset
(fill in details about the dataset you're using)
Dataset used was the Topic Modeling SetFit/20_newsgroups (https://huggingface.co/datasets/SetFit/20_newsgroups).

## Pre-trained Model
(fill in details about the pre-trained model you selected)

This is a BERTopic model. BERTopic is a flexible and modular topic modeling framework that allows for the generation of easily interpretable topics from large datasets (https://huggingface.co/guibvieira/topic_modelling). 

![image](https://github.com/user-attachments/assets/e8f8d756-38dd-4f16-9207-440a3c32db87)


## Performance Metrics
(fill in details about your chosen metrics and results)

Two metrics selected to evaluate performance of model:
Accuracy - indicates the overall percentage of correctly classified documents and provides fast baseline that is used for balanced classes but it is simple, and can potentially be misleading if some classes are more frequent.
F1 ('weighted')- F1 focuses on precision and recall and if they are high then F1 will be high. Like macro F1 score where provides the mean of precision and recall per class and averaged working well when used for imbalanced classes as it applies equal treatment across categories, but in weighted it classe by frequency. Output will focus on overall performance to reflect class imbalance. 

## Hyperparameters
(fill in details about which hyperparemeters you found most important/relevant while optimizing your model)

- removing extra spaces, removing punctuation/symbols, transition to lowercase

