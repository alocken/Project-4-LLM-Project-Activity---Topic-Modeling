# LLM Project
## Overall Project Task
Develop a Topic Modeling tool (NLP project using a pre-trained language model) for 'varied articles', that included: 
- selecting a topic and dataset,
- setting up project workspace,
- applying text processing and tokenization techniques,
- utilizing transfer learning for model development,
- evaluating and optimizing model performance, and
- completed final project report.

This project focused on topic classification (a specific type of text classification with goal to assign topic labels to text based content) using a transformer-based model. The goal was to classify short texts into meaningful topic categories using a combination of BERTopic for unsupervised clustering and a fine-tuned transformer model for supervised classification.

Improvements implemented in project:
-  Updated runtime type from CPU to GPU as required. This significantly improved duration of runtime and eliminated connectivity issues impacting training.
-  Uploading notebooks from Google Colab to GitHub presented multiple issues regarding metadata widgets which generated an 'Invalid Notebook'. To work around this issue, applied notebook cleaning code removing metadata widgets from all notebooks prior to uploading into GitHub. Although this cleaning code removed some visualization outputs, such as progress bars, it maintained the integrity of the code ran/critical outputs as those are still maintained in the cleaned notebook/GitHub upload.

## Dataset
The original dataset consisted of documents collected from SetFit/20_newsgroups (https://huggingface.co/datasets/SetFit/20_newsgroups)

Each document was clustered into a topic using BERTopic, which generates unsupervised topic labels. Topics with very few samples were filtered out to focus on high-quality clusters. The remaining data was split into train and validation sets using stratified sampling based on topic labels. The dataset used in this project consisted of a collection of text documents that were preprocessed and labeled using BERTopic, an unsupervised topic modeling technique. Each document represented a short text (such as a forum post, article snippet, or message), and the associated labels corresponded to topic IDs automatically assigned by BERTopic based on semantic similarity. To improve label quality and model performance, a class reduction step was applied where topics with fewer than 10 examples were filtered out to minimize noise and class imbalance. This resulted in a reduced, but more robust, set of topic labels. After filtering, the dataset was split into training and validation sets using stratified sampling to ensure proportional class representation. Each entry includes a cleaned text string and a numerical label representing a topic class, later mapped for compatibility with transformer models. The refined dataset enabled the training of a multi-class text classification model that predicts the most likely topic for new, unseen documents.

## Pre-trained Model
The pre-trained model selected is guibvieira/topic_modelling, a BERTopic-based model available on Hugging Face. BERTopic is a flexible and modular topic modeling framework that leverages transformer-based embeddings (e.g., MiniLM) along with clustering algorithms to generate interpretable topic representations from large collections of text. It excels at identifying coherent topic groups without requiring labeled data. In this project, BERTopic was used in the first phase to assign topic labels to an unlabeled dataset, effectively transforming the problem into a supervised classification task. These topic labels were then used as ground truth for training a classification model, enabling:

- Evaluation of classification performance on topic-prediction tasks
- Downstream usage such as topic-aware recommendations or content tagging
- Fine-tuning a transformer-based classification model (e.g., microsoft/MiniLM-L12-H384-uncased) to learn from the topic-labeled dataset

This hybrid approach combined unsupervised topic discovery with supervised learning, allowing for a more robust and scalable NLP solution.

(https://huggingface.co/guibvieira/topic_modelling). 

![image](https://github.com/user-attachments/assets/e8f8d756-38dd-4f16-9207-440a3c32db87)

The fine-tuning model used was microsoft/MiniLM-L12-H384-uncased, chosen for its architectural similarity to sentence-transformers/all-MiniLM-L6-v2, which was used for initial embedding generation. MiniLM offered a lightweight and efficient design, making it well-suited for scalable training and deployment. As a general-purpose transformer model pretrained on a large collection of short, topic-rich text documents, it adapts well to classification tasks even with moderately sized datasets. Fine-tuning was performed using Hugging Face's Trainer API to streamline training and evaluation workflows.

## Performance Metrics
The final model (Model #2) was evaluated using standard classification metrics including accuracy, weighted F1-score, and evaluation loss. A deicision was made, after a series of relativel low perforamnce, it was determined necessary to further fine tune Model #2 by applying class reduction, improving label consistency, and fine-tuning a more powerful transformer (bert-base-uncased). This decision generated an improved performance of the model as it achieved significantly better results compared to initial runs. The best evaluation outputs were:

- Evaluation Accuracy: ~42.2%
- Weighted F1-Score: ~0.31
- Evaluation Loss: ~2.91

These values marked a major improvement from early baselines where the model performed near random (accuracy ~8%, F1 ~0.03, loss ~6.0). The weighted F1-score was especially important due to the presence of class imbalance in the BERTopic-generated labels.

## Hyperparameters
The final model (Model #2) was further fine tuned, from the fine tuning completed in the earlier baselines, using an improved set of hyperparameters optimized for stability and generalization in a multi-class, class-imbalanced setting. These included:

- Learning Rate: 1e-5; lowered from earlier runs to allow more stable, slower convergence
- Epochs: 6; increased from 3 to give the model more opportunity to learn from the data
- Batch Size: 8 per device for both training and evaluation; smaller batch size to encourage better generalization
- Weight Decay: 0.01; regularization to reduce overfitting
- Warmup Steps: 100; gradual increase of the learning rate at the beginning to avoid training instability
- Gradient Clipping (max_grad_norm): 1.0; prevent exploding gradients and ensure smoother updates
- Max Sequence Length: 128 tokens; aligned with the tokenizer’s expected input and suitable for the average document length
- Metric for Best Model: f1_weighted; prioritizes performance under class imbalance

These changes, designed in response to early model underperformance, were combined with label filtering and a switch to the more expressive bert-base-uncased model for improved results. These adjustments resulted in significant improvements overall.

## Potential Next Steps
The next phase would involve extending the model's applicability beyond the training environment. This includes evaluating its performance on new or real-world text samples. This process would leverage the deployment strategy and address potential ethical considerations detailed in notebook 5_Deployment. Once deployed, it would be important to monitor predictions in production, track performance over time, and support iterative fine-tuning based on user feedback and evolving data.

## Reference: Final Project 4 Link - Model #2 on Hugging Face 
alocken/topic_modeling_project4 (https://huggingface.co/alocken/topic_modeling_project4)
![image](https://github.com/user-attachments/assets/9b67f4af-5a8b-40c2-8aff-44b48242a6c2)
