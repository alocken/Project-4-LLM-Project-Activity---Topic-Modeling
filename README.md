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

Improvements implemented in project:
-  Observed that connectivty impacted ability to complete the full runtime required to complete the train. In response, updated runtime type from CPU to GPU. This significantly improved duration of runtime and eliminated connectivity issues impacting training.
-  Implemented check points saves mounted to Google Drive folder to improve ability to retrieve training if connectivity resulted in issues and loss of progress on train.
-  
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
- Accuracy - indicates the overall percentage of correctly classified documents and provides fast baseline that is used for balanced classes but it is simple, and can potentially be misleading if some classes are more frequent. Measures the proportion of correct answers provided by the model.
- F1 - focuses on precision and recall and if they are high, then F1 will be high. Measures model accuracy in classifcication tasks, considering both precision (number correct positive results divided by number of all positive results) and recall (number of correct positive results divided by number of positive results that should have been identified).
- Precision - measureing the proportion of correctly predicted positive observations out of all predicted positive observations. Ideal value is close to 1.
- Recall - proportion of correctly predicted positive observations out of all actual positive observations. Ideal value is close to 1.

Outputs Summary:from epoch 1 to 2, training loss decreased (1.68 to 1.00), validation loss decerased (1.12 to 1.05), and all metrics (accuracy, F1, precision, recall) improved. Model is learning well and generalizing better. The loss and metrics indicate good overall learning progress. 

Extra step: 
Added in an extra step in model training as was curious about outputs if reduced epoch and increased steps to embed evaluation during training. Outputs identified F1 as the measurement for 'best' model. Although not saved as the final model, it was an interesting comparative exercise to demonstrate and visualize the different outputs. This additional step has been included in notebook.

## Hyperparameters
(fill in details about which hyperparemeters you found most important/relevant while optimizing your model)

- removing extra spaces, removing punctuation/symbols, transition to lowercase

