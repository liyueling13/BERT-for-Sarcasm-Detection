# BERT-for-Sarcasm-Detection
Figurative language like irony and sarcasm can be quite difficult to detect, even by humans. It doesn’t mean what it says, it signifies something different than what it means. It is “the expression of one's meaning by using language that normally signifies the opposite, typically for humorous or emphatic effect” (Oxford Languages Dictionary). It’s a kind of figurative/poetic language. It requires deep context, abstract understanding, nuance and precision, a sense of humor, and more. To that, I wanted to train a deep learning model to detect irony and/or sarcasm.

## Data
I found two datasets: 
1. The first, from Hugging Face, contained [tweets labelled for irony]([url](https://huggingface.co/datasets/tweet_eval)) 
2. The second, from Princeton, contains [Reddit comments labelled for sarcasm]([url](https://nlp.cs.princeton.edu/old/SARC/1.0/main/)), [also paper here]([url](https://arxiv.org/pdf/1704.05579.pdf)https://arxiv.org/pdf/1704.05579.pdf).

## Methods
1. Connected to Google Colab and Kaggle's TPUs to be able to run my model efficiently
2. To load my datasets, used Hugging Face datasets package and mounted and imported Google Colab Drive
3. Learned about transformer models: encodings, attention masks, and self-attention
4. Used pandas to clean and wrangle data
5. Learned both Distilbert and BERT models in Tensorflow and Pytorch (tokenized data, loaded into TF datasets vs. torch dataloaders, froze base layers)
6. Wrote a custom classification head with more deep learning layers for BertForSequenceClassification
7. Set parameters for my models including optimizers, learning rates, loss functions (BinaryCrossentropy), epochs, early stopping, metrics, and batch size

## Results
Unfortunately, I was never able to train a model that returned more than 50% accuracy. Looking at others who have attempted the same project, however--no one has been able to succeed so far. If I wanted to try again, I would have to significantly modify scope/setup of my question: e.g. as with the Princeton Sarcasm paper, I would need to provide one prompt and two responses (ironic or not, sarcastic or not). Since I don't have such a dataset, I chose to pause this project for now (after much productive, if fruitless learning!).
