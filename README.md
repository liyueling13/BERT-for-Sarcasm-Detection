# BERT-for-Sarcasm-Detection
Figurative language like irony and sarcasm can be quite difficult to detect. It doesn’t mean what it says; it signifies something different than what it means. It is 
> “the expression of one's meaning by using language that normally signifies the opposite, typically for humorous or emphatic effect” (Oxford Languages Dictionary).

It requires deep context, abstract understanding, nuance and precision, a sense of humor, and more. 

I may not even be able to tell when someone is being ironic or sarcastic; can a LLM detect whether or not a comment is ironic or sarcastic? Let's fine-tune a BERT model to detect irony and sarcasm.

## Data
I found two datasets: 
1. The first, from Hugging Face, contained [tweets labelled for irony](https://huggingface.co/datasets/tweet_eval) 
2. The second, from Princeton NLP, contains [Reddit comments labelled for sarcasm](https://nlp.cs.princeton.edu/old/SARC/1.0/main/), ![see also paper here](https://arxiv.org/pdf/1704.05579.pdf).

## Methods
1. Connected to Google Colab and Kaggle's TPUs to be able to run my model efficiently
2. To load my datasets, used Hugging Face datasets package and mounted and imported Google Colab Drive
3. Learned about transformer models: encodings, attention masks, and self-attention
4. Used pandas to clean and wrangle data: see [Sarcasm - EDA](https://github.com/liyueling13/BERT-for-Sarcasm-Detection/blob/main/Sarcasm%20-%20EDA.ipynb)
5. Learned both Distilbert and BERT models in Tensorflow and Pytorch (tokenized data, loaded into TF datasets vs. torch dataloaders, froze base layers): see [Irony - Distilbert](https://github.com/liyueling13/BERT-for-Sarcasm-Detection/blob/main/Irony%20-%20Distilbert%20model.ipynb) and [Sarcasm - BERT](https://github.com/liyueling13/BERT-for-Sarcasm-Detection/blob/main/Sarcasm%20-%20BERT%20model.ipynb)
6. Wrote a custom classification head with more deep learning layers for BertForSequenceClassification
7. Set parameters in Tensorflow for my models including optimizers, learning rates, loss functions (BinaryCrossentropy), epochs, early stopping, metrics, and batch size

## Results
Unfortunately, I was never able to train a model that returned more than 50% accuracy on the validation set. There was also very little documentation for BERT in Tensorflow and PyTorch on Hugging Face, which made it difficult to look at the model architecture in more detail to debug my model.

Even my data science mentor was not able to achieve a better result. Looking at others on Google/Github who have attempted the same project with the Princeton NLP sarcasm dataset, no one has been able to succeed so far.

If I wanted to try again, I would have to significantly modify scope/setup of my question. For instance, the authors of Princeton Sarcasm paper trained their model in a completely different way (not provided in their dataset). 

> "We construct a balanced learning task by taking only one sarcastic and one non-sarcastic response from each set of responses to a comment sequence. The task then becomes one of picking which of two statements that share a context is sarcastic, with performance measured by accuracy..."

Since I don't have such a dataset, I chose to pause this project for now.

## Future Steps
When I have time, I would like to fine-tune a BERT model [to predict banned books]([url](https://github.com/liyueling13/Banned-Books-with-Topic-Modelling-and-Logistic-Regression/)https://github.com/liyueling13/Banned-Books-with-Topic-Modelling-and-Logistic-Regression/).
