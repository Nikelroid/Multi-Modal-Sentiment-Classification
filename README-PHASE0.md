<h1>MSCTD (Multi-Modal Sentiment Classification and Time Dynamics)</h1>

<p>MSCTD is a tool designed for multi-modal sentiment analysis and time dynamics exploration in image-text conversations. It provides functionalities to preprocess data, create datasets, perform sentiment analysis, visualize sentiments, analyze text lengths, and explore time patterns within conversations.</p>

<h2>Phase 0: Define Dataloader</h2>

<p>In this phase, we prepare a appropriate data loader for load data from memory. This class can get used in next phases.</p>

<h2>Getting Started</h2>

<p>To use MSCTD, follow these steps:</p>

<ol>
  <li>Mount Google Drive to your Colab notebook:</li>
  <pre><code>from google.colab import drive
drive.mount('/content/drive')
  </code></pre>
  <li>Clone the MSCTD repository:</li>
  <pre><code>!git clone https://github.com/XL2248/MSCTD
  </code></pre>
  <li>Install dependencies:</li>
  <pre><code>!pip install --upgrade --no-cache-dir gdown mtcnn
  </code></pre>
  <li>Download necessary files:</li>
  <pre><code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj
!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W
  </code></pre>
  <li>Unzip the downloaded files:</li>
  <pre><code>%%bash
unzip -qq train_ende.zip
unzip -qq test.zip
  </code></pre>
  <li>Organize the dataset:</li>
  <pre><code>!mkdir dataset
!cd dataset; mkdir train test dev
!mv *train* dataset/train
!mv *test* dataset/test
!mv *dev* dataset/dev
  </code></pre>
</ol>

<h2>Usage</h2>

<ol>
  <li>Import the necessary modules:</li>
  <pre><code>import torch
from torchvision import transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
  </code></pre>
  <li>Define your dataset:</li>
  <pre><code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt')
  </code></pre>
  <li>Access individual data points:</li>
  <pre><code>text, image, sentiment = trainset[14787].values()
  </code></pre>
</ol>

<h2>Features</h2>

<ul>
  <li>Multi-modal sentiment analysis</li>
  <li>Data preprocessing utilities</li>
  <li>Visualization tools for sentiment analysis and time patterns</li>
</ul>

<h2>Contributing</h2>

<p>Contributions to MSCTD are welcome! Here's how you can contribute:</p>

<ol>
  <li>Fork the repository</li>
  <li>Create your feature branch (<code>git checkout -b feature/YourFeature</code>)</li>
  <li>Commit your changes (<code>git commit -am 'Add some feature'</code>)</li>
  <li>Push to the branch (<code>git push origin feature/YourFeature</code>)</li>
  <li>Create a new Pull Request</li>
</ol>
