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

Note: uploading notebooks from Google Colab to GitHub presented issues regarding metadata widgets which generated an 'Invalid Notebook'. To work around this error and issue, applied code to remove the metadata widgets and maintain the outputs. This generated a clean notebook in Colab that was successfully loaded into GitHub but noted in some notebooks (e.g., 3-pre-trained model) that some of preprocess code indicates 0% but was in fact complete as supported by the continued outputs in the code following. Confirmed the outputs and evalutaions are correct.

## Dataset
(fill in details about the dataset you're using)
Dataset used was the Topic Modeling SetFit/20_newsgroups (https://huggingface.co/datasets/SetFit/20_newsgroups).

## Pre-trained Model
(fill in details about the pre-trained model you selected)

This is a BERTopic model. BERTopic is a flexible and modular topic modeling framework that allows for the generation of easily interpretable topics from large datasets (https://huggingface.co/guibvieira/topic_modelling). 

![image](https://github.com/user-attachments/assets/e8f8d756-38dd-4f16-9207-440a3c32db87)


## Performance Metrics
(fill in details about your chosen metrics and results)

Metrics selected to evaluate performance of model:
Accuracy - indicates the overall percentage of correctly classified documents and provides fast baseline that is used for balanced classes but it is simple, and can potentially be misleading if some classes are more frequent.
F1 - focuses on precision and recall and if they are high, then F1 will be high. 
Precision - measureing the proportion of correctly predicted positive observations out of all predicted positive observations. Ideal value is close to 1.
Recall - proportion of correctly predicted positive observations out of all actual positive observations. Ideal value is close to 1.
Roc Auc (Receiver Operating Characteristic - Area Under Curve) - measures how well the model distinguishes between positive and negative classes. Plots true positive rate (recall) against false positive rate at various thresholds: 1=perfect classification, .5=not better than random guessing, <.5=worse than random (indicates something wrong). Ideal value is close to 1.

F1 selectd to determine best model at end of train.

During train, performed one epoch, reduced from two, due to time required to run two (14 hours versus 7 hoursa. Also, experienced issues with connectivity interupting runtime and in response increased the frequency of checkpoints to save every 15 steps to enure storing the most recent 3 saves in the event the runtime was interupted mid-run. This ensured progress would not be lost and coudl be restored to the most recent checkpoint as/if required.

## Hyperparameters
(fill in details about which hyperparemeters you found most important/relevant while optimizing your model)

- removing extra spaces, removing punctuation/symbols, transition to lowercase

