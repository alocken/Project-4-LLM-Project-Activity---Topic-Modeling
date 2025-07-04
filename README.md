# LLM Project
## Overall Project Task
Develop a Topic Modeling tool (NLP project using a pre-trained language model) for 'varied articles', that included: 
- selecting a topic and dataset,
- setting up project workspace,
- applying text processing and tokenization techniques,
- utilizing transfer learning for model development,
- evaluating and optimizing model performance, and
- completed final project report.

This project focused on topic classification (a specific type of text classification with goal to assign topic labels to text based content) using a transformer-based model. The goal is to classify short texts into meaningful topic categories using a combination of BERTopic for unsupervised clustering and a fine-tuned transformer model for supervised classification.

Improvements implemented in project:
-  Updated runtime type from CPU to GPU as required. This significantly improved duration of runtime and eliminated connectivity issues impacting training.
-  Uploading notebooks from Google Colab to GitHub presented multiple issues regarding metadata widgets which generated an 'Invalid Notebook'. To work around this error and issue, applied notebook cleaning code removing metadata widgets. Although this cleaning code removed visualization outputs such as progress bars on code but did not change/impact integrity of the overall notebook outputs/outcomes as they remain correct and in tact. 

## Dataset
The dataset consisted of text documents collected from SetFit/20_newsgroups (https://huggingface.co/datasets/SetFit/20_newsgroups)
Each document was clustered into a topic using BERTopic, which generates unsupervised topic labels. Topics with very few samples were filtered out to focus on high-quality clusters. The remaining data was split into train and validation sets using stratified sampling based on topic labels.

## Pre-trained Model
The pre-trained model selected is guibvieira/topic_modelling, a BERTopic-based model available on Hugging Face. BERTopic is a flexible and modular topic modeling framework that leverages transformer-based embeddings (e.g., MiniLM) along with clustering algorithms to generate interpretable topic representations from large collections of text. It excels at identifying coherent topic groups without requiring labeled data.
In this project, BERTopic was used in the first phase to assign topic labels to an unlabeled dataset, effectively transforming the problem into a supervised classification task. These topic labels were then used as ground truth for training a classification model, enabling:
- Evaluation of classification performance on topic-prediction tasks
- Downstream usage such as topic-aware recommendations or content tagging
- Fine-tuning a transformer-based classification model (e.g., microsoft/MiniLM-L12-H384-uncased) to learn from the topic-labeled dataset

This hybrid approach combined unsupervised topic discovery with supervised learning, allowing for a more robust and scalable NLP solution.

(https://huggingface.co/guibvieira/topic_modelling). 

![image](https://github.com/user-attachments/assets/e8f8d756-38dd-4f16-9207-440a3c32db87)

The fine-tuning model used in this project is microsoft/MiniLM-L12-H384-uncased, chosen for its architectural similarity to sentence-transformers/all-MiniLM-L6-v2, which was used for initial embedding generation. MiniLM offers a lightweight and efficient design, making it well-suited for scalable training and deployment. As a general-purpose transformer model pretrained on large corpora, it adapts well to classification tasks even with moderately sized datasets. Fine-tuning was performed using Hugging Face’s Trainer API to streamline training and evaluation workflows.

## Performance Metrics
To evaluate the model’s effectiveness, several standard metrics were used. These included accuracy, weighted F1-score, and evaluation loss. In a typical training run, the model achieved an accuracy of approximately 0.81, a weighted F1-score of 0.78, and an evaluation loss of 0.45. The weighted F1-score was particularly emphasized in this project due to class imbalance in the topic labels, where some categories were significantly more frequent than others. These metrics provide a well-rounded view of how well the model handles multi-class classification under imbalanced data conditions. Final values may vary depending on dataset size and label filtering.

## Hyperparameters
The model was fine-tuned using a set of carefully selected hyperparameters to balance performance and resource efficiency. A learning rate of 2e-5 was used, as it is a common and effective starting point for transformer-based models. Training was run for 3 epochs, which provided a good trade-off between underfitting and overfitting. The batch size was set to 16, optimized for GPU memory constraints and to maintain stable gradient updates. A weight decay of 0.01 was included as a form of regularization to reduce overfitting. The maximum sequence length was capped at 128 tokens, as the majority of text samples in the dataset were relatively short and did not require longer input lengths.

For future optimization, further hyperparameter could be explored to find more optimal configurations. Additionally, experimenting with alternative lightweight transformers, such as distilbert, may result in further improvements in efficiency or accuracy.

## Potential Next Steps
The next phase would involve extending the model's applicability beyond the training environment. This includes evaluating its performance on new or real-world text samples. This process would leverage the deployment strategy and address potential ethical considerations detailed in notebook 5_Deployment. Once deployed, it would be important to monitor predictions in production, track performance over time, and support iterative fine-tuning based on user feedback and evolving data.
