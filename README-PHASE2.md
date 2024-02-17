<h1>PHASE 2: Enhancing Multi-Modal Sentiment Analysis with Text Analysis</h1>

<h2>Overview</h2>
<p>In Phase 2 of the project, we delve into the realm of natural language processing (NLP), augmenting our multi-modal sentiment analysis pipeline. This phase encompasses the development of code for the text analysis component of the project. Here, we focus on training models to extract and process textual data (dialogs) from the dataset. Subsequently, we preprocess the text data to make it suitable for training, followed by the classification of text into three sentiment classes: positive, negative, and neutral. The integration of text analysis significantly enhances the functionality of the project, laying a strong foundation for subsequent phases. For the final results of this phase, please refer to Phase 3 of the project.</p>

<h2>Contents</h2>

<ol>
  <li><a href="#setup">Setup</a></li>
  <li><a href="#data-preparation">Data Preparation</a></li>
  <li><a href="#dataset-and-dataloader">Dataset and Dataloader</a></li>
  <li><a href="#graphs-and-analysis">Graphs and Analysis</a></li>
  <li><a href="#phase-2">Phase 2</a></li>
</ol>

<hr>

<h3 id="setup">1. Setup</h3>
<p>Before running any code, make sure to set up the necessary environment. We recommend using Google Colab for running the code, as it provides convenient access to GPU resources. Here are the steps to set up the environment:</p>

<pre><code>from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/XL2248/MSCTD

import os
import shutil

for file in os.listdir('MSCTD/MSCTD_data/ende'):
    if file.startswith('english_'):
        shutil.copy('MSCTD/MSCTD_data/ende/' + file, file)
    if file.startswith('image_index_'):
        shutil.copy('MSCTD/MSCTD_data/ende/' + file, file)
    if file.startswith('sentiment_'):
        shutil.copy('MSCTD/MSCTD_data/ende/' + file, file)

!pip install --upgrade --no-cache-dir gdown
!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj
!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W
%%bash
for x in *.zip
do
  unzip -qq $x
done;

os.makedirs('dataset', exist_ok=True)
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)
os.makedirs('dataset/dev', exist_ok=True)

for file in os.listdir():
    if 'train' in file:
        shutil.move(file, 'dataset/train')
    if 'test' in file:
        shutil.move(file, 'dataset/test')
    if 'dev' in file:
        shutil.move(file, 'dataset/dev')
</code></pre>

<h3 id="data-preparation">2. Data Preparation</h3>
<p>After setting up the environment, the next step is to prepare the data. The provided code downloads the necessary data files and organizes them into the <code>dataset</code> directory.</p>

<pre><code>...
</code></pre>

<h3 id="dataset-and-dataloader">3. Dataset and Dataloader</h3>
<p>The <code>MSCTD_Dataset</code> class defined in the code handles the dataset loading and preprocessing. It loads images, text, and sentiment labels from the dataset directory and provides methods for accessing the data.</p>

<pre><code>...
</code></pre>

<h3 id="graphs-and-analysis">4. Graphs and Analysis</h3>
<p>The code includes various analysis and visualization techniques for understanding the dataset better. This includes histograms of sentiments, length of sentences, correlation between sentiments and sentence lengths, histograms of image counts in conversations, histograms of face counts in images, and analysis of time patterns.</p>

<pre><code>...
</code></pre>

<h3 id="phase-2">5. Phase 2</h3>
<p>Phase 2 of the code focuses on text preprocessing and classification using TF-IDF vectors. It includes preprocessing steps such as tokenization, stopword removal, lemmatization, handling numbers and unknown words. Additionally, it implements a <code>Text_MSCTD</code> class for handling text data separately. Finally, it performs classification using TF-IDF vectors with logistic regression and SVM classifiers.</p>

<pre><code>...
</code></pre>

<hr>

<p>Feel free to explore the code further and adapt it to your specific needs. If you have any questions or issues, please don't hesitate to reach out. Happy coding!</p>

</body>
</html>
