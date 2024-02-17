<h1>PHASE 2: Enhancing Multi-Modal Sentiment Analysis with Text Analysis</h1>

<h2>Overview</h2>
<p>In Phase 2 of the project, we delve into the realm of natural language processing (NLP), augmenting our multi-modal sentiment analysis pipeline. This phase encompasses the development of code for the text analysis component of the project. Here, we focus on training models to extract and process textual data (dialogs) from the dataset. Subsequently, we preprocess the text data to make it suitable for training, followed by the classification of text into three sentiment classes: positive, negative, and neutral. The integration of text analysis significantly enhances the functionality of the project, laying a strong foundation for subsequent phases. For the final results of this phase, please refer to Phase 3 of the project.</p>

<h2>Contents</h2>

<p>1- Loading dataset using dataloader</p>
<p>2- Preprocessing text data</p>
<p>3- Train by TF-IDF</p>
<p>4- Test and evaluation TF-IDF model</p>
<p>5- Use svm in TF-IDF</p>
<p>6- Train by Pseudo Word2Vec</p>
<p>7- Test and evaluation Word2Vec model</p>
<p>8- Using pretrained model</p>
<p>9- Using pretrained BERT</p>
<p>10- Train Pretraied</p>
<p>11- Evaluate Pretrained</p>

<h2>Part 1: Loading dataset using DataLoader</h2>
<p>
    The dataset is loaded using a custom DataLoader class, <code>Text_MSCTD</code>, which inherits from <code>MSCTD_Dataset</code>. This class preprocesses the text data and allows for easy handling of the dataset during training and testing.
</p>
<pre><code>
# We will change our Dataset class a bit in this part, to avoid the overhead of loading images.
class Text_MSCTD(MSCTD_Dataset):
    def __init__(self, dataset_dir, conversation_dir, texts, sentiments,
                preprocess_func=None, pad_idx=None, max_len=None, transform=None, images_dir=''):
        super().__init__(dataset_dir, images_dir, conversation_dir, texts, sentiments, transform)
        self.preprocess_func = preprocess_func
        self.pad_idx = pad_idx
        self.max_len = max_len

    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.preprocess_func is not None:
            text = self.preprocess_func(text)
            if self.max_len is not None:
                text = text[:self.max_len]
            if self.pad_idx is not None:
                text = F.pad(torch.tensor(text), (0, self.max_len - len(text)), 'constant', self.pad_idx)
        labels = self.sentiments[idx]
        return text, labels
train_dataset = Text_MSCTD('dataset/train', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt')
print(train_dataset[10])
print(sent_preprocess(train_dataset[10][0]))
dev_dataset = Text_MSCTD('dataset/dev', 'image_index_dev.txt', 'english_dev.txt', 'sentiment_dev.txt')
test_dataset = Text_MSCTD('dataset/test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt')
</code></pre>


<h2>Part 2: Preprocessing text data</h2>
<p>
    Text data is preprocessed using various techniques such as lowercasing, removing punctuation and stopwords, lemmatization, handling numbers, and handling unknown words.
</p>
<pre><code>
# Phase 2
!pip install pyenchant
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

import enchant
english_dict = enchant.Dict("en_US")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import torch
import torch.nn as nn
import torch.nn.functional as F
NUM = '&lt;NUM&gt;'
UNK = '&lt;UNK&gt;'

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def sent_preprocess(sent, lower=True, remove_punct=True, remove_stopwords=True,
                    lemmatize=True, handle_nums=True, handle_unknowns=True):
    if lower:
        sent = sent.lower()
    if remove_punct:
        sent = sent.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(sent)
    if remove_stopwords:
        word_tokens = [w for w in word_tokens if not w in stop_words]
    if lemmatize:
        word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
    if handle_nums:      
        def is_number(s):
            if s.isdigit():
                return True
            if s[:-2].isdigit():
                if s[-2:] == 'th' or s[-2:] == 'st' or s[-2:] == 'nd' or s[-2:] == 'rd':
                    return True
            return False
        word_tokens = [NUM if is_number(w) else w for w in word_tokens]
    if handle_unknowns:
        word_tokens = [w if english_dict.check(w) else UNK for w in word_tokens]
    return word_tokens
</code></pre>


<h2>Part 3: Train by TF-IDF</h2>
<p>
    The TF-IDF model is trained on the preprocessed text data using <code>TfidfVectorizer</code> from scikit-learn.
</p>
<pre><code>
## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=sent_preprocess)
tfidf_data = tfidf.fit_transform(all_texts)
tfidf_data = tfidf_data.toarray()
tfidf.get_feature_names_out()
print(len(all_texts))
print(tfidf_data.shape)
</code></pre>


<h2>Part 4: Test and evaluation TF-IDF model</h2>
<p>
    The trained TF-IDF model is tested and evaluated on the test set, with metrics such as accuracy, precision, recall, and F1-score calculated.
</p>
<pre><code>
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

average_policy = 'macro'
metrics = {'accuracy': accuracy_score, 'precision': lambda y1, y2: precision_score(y1, y2, average=average_policy),
           'recall': lambda y1, y2: recall_score(y1, y2, average=average_policy),
           'f1': lambda y1, y2: f1_score(y1, y2, average=average_policy),
           'confusion_matrix': confusion_matrix}
plt.figure(figsize=(10, 10))

def eval(y_true, y_pred, metrics=metrics, plot_confusion_matrix=True):
    metrics_values = {name: metric(y_true, y_pred) for name, metric in metrics.items()}
    if plot_confusion_matrix:
        ConfusionMatrixDisplay(metrics_values.pop('confusion_matrix')).plot()
    return metrics_values

# Train results:
eval(all_labels, model.predict(tfidf_data))

# Test results:
test_tfidf_data = tfidf.transform(test_text).toarray()
eval(test_labels, model.predict(test_tfidf_data))
</code></pre>


 <h2>Part 5: Use SVM in TF-IDF</h2>
<p>
    In this part, SVM (Support Vector Machine) classifier is applied to the TF-IDF vectors for classification.
</p>
<pre><code>
from sklearn import svm

model = Pipeline([('pca', PCA(n_components=100)), ('svm', svm.SVC())])
model.fit(tfidf_data, all_labels)
eval(all_labels, model.predict(tfidf_data))
eval(test_labels, model.predict(test_tfidf_data))
</code></pre>


<h2>Part 6: Train by Pseudo Word2Vec</h2>
<p>
    Pseudo Word2Vec embeddings are generated by training an SVM classifier on TF-IDF vectors and using the coefficients as word embeddings.
</p>
<pre><code>
## Pseudo Word2Vec!
from sklearn.base import TransformerMixin

# to convert sparse matrix to dense
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return np.array(X.todense())
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA

all_sents = [sent_preprocess(text, remove_stopwords=False) for text in all_texts]
reduced_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer=lambda x: x)),
    ('to_dense', DenseTransformer()),
    ('pca', PCA(n_components=50)),
    ('prep', preprocessing.StandardScaler())
])
reduced_tfidf.fit(all_sents)
tfidf = reduced_tfidf[0]
all_words = tfidf.get_feature_names_out()
word2idx = {word: idx for idx, word in enumerate(all_words)}
idx2word = {idx: word for idx, word in enumerate(all_words)}
vocab_size = len(all_words)
vocab_size
UNK_IDX = word2idx[UNK]

def text_to_idxs(text):
    return [word2idx.get(word, UNK_IDX) for word in text]

def idxs_to_text(idxs):
    return [idx2word[idx] for idx in idxs]

all_sent_idxs = [text_to_idxs(sent) for sent in all_sents]
word_embeddings = {idx: [] for idx in range(vocab_size)}

from tqdm import tqdm

num_epochs = 1

for epoch in range(num_epochs):
    for sent in tqdm(all_sent_idxs, total=len(all_sent_idxs), desc=f'Epoch {epoch}'):
        words = set(sent)
        for word in words:
            # positive_samples = list(words - {word})
            positive_samples = list(words)
            num_samples = len(positive_samples)
            negative_samples = np.random.choice(vocab_size, num_samples, replace=False)
            data_idx = np.concatenate((positive_samples, negative_samples))
            data = reduced_tfidf.transform([idx2word[idx] for idx in data_idx])
            labels = np.concatenate((np.ones(num_samples), np.zeros(num_samples)))
            svc = svm.SVC(kernel='linear')
            svc.fit(data, labels)
            word_embeddings[word].append(np.concatenate((svc.coef_[0], svc.intercept_)))

for word, embeddings in word_embeddings.items():
    word_embeddings[word] = np.mean(embeddings, axis=0)
</code></pre>


<h2>Part 7: Test and evaluation Word2Vec model</h2>
<p>
    The Word2Vec model is evaluated by using the word embeddings generated from TF-IDF vectors to represent sentences, and then training a classifier on these representations for sentiment classification.
</p>
<pre><code>
Now we will use the embedding we have found for the words, to train a model to predict the sentiment of the sentence.  
Note that we need an embedding for the sentence to be able to do that. So we will use the embedding of the words in the sentence, and average them to get the embedding of the sentence.
sent_embeddings = [np.mean([word_embeddings[word] for word in sent], axis=0) for sent in all_sent_idxs]
model = Pipeline([('prep', preprocessing.StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
model.fit(sent_embeddings, all_labels)
# eval on train
eval(all_labels, model.predict(sent_embeddings))
# eval on test
test_sent_idxs = [text_to_idxs(sent_preprocess(text, remove_stopwords=False)) for text in test_text]
test_sent_embeddings = [np.mean([word_embeddings[word] for word in sent], axis=0) for sent in test_sent_idxs]

eval(test_labels, model.predict(test_sent_embeddings))
</code></pre>


<h2>Part 8: Using pretrained model</h2>
<p>
    In this part, pretrained word embeddings from GloVe are loaded and used to represent words in the dataset.
</p>
<pre><code>
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

word2vec = {}
for line in lines:
    line = line.split()
    word = line[0]
    vec = np.array(line[1:], dtype=np.float32)
    word2vec[word] = vec
</code></pre>


<h2>Part 9: Using pretrained BERT</h2>
<p>
    Pretrained BERT (Bidirectional Encoder Representations from Transformers) model is utilized for text classification tasks.
</p>
<pre><code>
!pip install transformers
from transformers import BertForSequenceClassification

# Load the pretrained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 3,  # Number of labels for sentiment classification
    output_attentions = False,
    output_hidden_states = False,
)
model.to(device);  # Transfer the model to the available device (GPU or CPU)
</code></pre>


<h2>Part 10: Train Pretrained</h2>
<p>
    The pretrained BERT model is trained on the dataset for text classification.
</p>
<pre><code>
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Train the model
model.train()
for epoch in range(EPOCHS):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
</code></pre>


<h2>Part 11: Evaluate Pretrained</h2>
<p>
    The pretrained BERT model is evaluated on the test set to assess its performance for text classification.
</p>
<pre><code>
# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Evaluate the model on the test set
test_accuracy = evaluate(model, test_loader)
print(f'Test Accuracy: {test_accuracy}')
</code></pre>

</body>
</html>

