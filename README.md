# BERT-for-Sarcasm-Detection
Figurative language like irony and sarcasm can be quite difficult to detect, even by humans. It doesn’t mean what it says; it signifies something different than what it means. It is “the expression of one's meaning by using language that normally signifies the opposite, typically for humorous or emphatic effect” (Oxford Languages Dictionary). It’s a kind of figurative/poetic language. It requires deep context, abstract understanding, nuance and precision, a sense of humor, and more. To that, I wanted to train a deep learning model to detect irony and/or sarcasm.

## Data
I found two datasets: 
1. The first, from Hugging Face, contained [tweets labelled for irony]([url](https://huggingface.co/datasets/tweet_eval)) 
2. The second, from Princeton NLP, contains [Reddit comments labelled for sarcasm]([url](https://nlp.cs.princeton.edu/old/SARC/1.0/main/)), [see also paper here]([url](https://arxiv.org/pdf/1704.05579.pdf)https://arxiv.org/pdf/1704.05579.pdf).

## Methods
1. Connected to Google Colab and Kaggle's TPUs to be able to run my model efficiently
2. To load my datasets, used Hugging Face datasets package and mounted and imported Google Colab Drive
3. Learned about transformer models: encodings, attention masks, and self-attention
4. Used pandas to clean and wrangle data
5. Learned both Distilbert and BERT models in Tensorflow and Pytorch (tokenized data, loaded into TF datasets vs. torch dataloaders, froze base layers)
6. Wrote a custom classification head with more deep learning layers for BertForSequenceClassification
7. Set parameters in Tensorflow for my models including optimizers, learning rates, loss functions (BinaryCrossentropy), epochs, early stopping, metrics, and batch size

## Results
Unfortunately, I was never able to train a model that returned more than 50% accuracy on the validation set. Even my data science mentor was not able to achieve a better result! There was also very little documentation for BERT in Tensorflow and PyTorch on Hugging Face, which made it difficult to look at the model architecture in more detail to debug my model.

Looking at others on Google/Github who have attempted the same project with the Princeton NLP sarcasm dataset, however--no one has been able to succeed so far. Those who have attempted the project with LLMs have given up, and/or achieved reasonable results using the same training set data (vs. validation/test/holdout data).

If I wanted to try again, I would have to significantly modify scope/setup of my question. For instance, the authors of Princeton Sarcasm paper trained their model in a completely different way (not provided in their dataset). 

"We construct a balanced learning task by taking only one sarcastic and one non-sarcastic response from each set of responses to a comment sequence. The task then becomes one of picking which of two statements that share a context is sarcastic, with performance measured by accuracy..."

Since I don't have such a dataset, I chose to pause this project for now (after much productive learning, although any discernable result to show for it).
