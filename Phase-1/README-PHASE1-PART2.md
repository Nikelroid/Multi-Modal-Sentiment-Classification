<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
<!-- Mount Google Drive -->
<div>
<h2>Mount Google Drive</h2>
<p>This code mounts Google Drive to the Colab environment. It's necessary for accessing files stored in Google Drive.</p>
<pre><code>from google.colab import drive</code></pre>
<pre><code>drive.mount('/content/drive')</code></pre>
</div>

<!-- Change Directory -->
<div>
<h2>Change Directory</h2>
<p>This section changes the directory to the project directory and performs initial setup tasks like copying files, cloning repositories, and unzipping datasets.</p>
<pre><code>%cd drive/My Drive/deep_learning/Project</code></pre>
<em>Run this just for the first time:</em><br>
<pre><code>!cp train_ende.zip .</code></pre>
<pre><code>!cp test.zip .</code></pre>
<pre><code>!git clone https://github.com/XL2248/MSCTD</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/english_*.txt .</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/image_index_*.txt .</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .</code></pre>
<pre><code>!pip install --upgrade --no-cache-dir gdown</code></pre>
<pre><code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj</code></pre>
<pre><code>!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W</code></pre>
<pre><code>%%bash</code></pre>
<pre><code>for x in dataset/*.zip</code></pre>
<pre><code>do</code></pre>
<pre><code>  unzip -qq $x</code></pre>
<pre><code>done</code></pre>
<pre><code>!mkdir dataset</code></pre>
<pre><code>!cd dataset; mkdir train test dev</code></pre>
<pre><code>!mv *train* dataset/train</code></pre>
<pre><code>!mv *test* dataset/test</code></pre>
<pre><code>!mv *dev* dataset/dev</code></pre>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Dataset and Dataloader</h2>
<p>This part involves setting up the dataset and dataloader for training and testing the model.</p>
<pre><code>import torch</code></pre>
<pre><code>from torchvision import transforms as T</code></pre>
<pre><code>from PIL import Image</code></pre>
<pre><code>import os</code></pre>
<pre><code>from pathlib import Path</code></pre>
<pre><code>import numpy as np</code></pre>
<pre><code>from torch.utils.data import Dataset, DataLoader</code></pre>
<pre><code>import torchvision.transforms as transforms</code></pre>
<pre><code>transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((288,288), interpolation=Image.BICUBIC),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])</code></pre>
<pre><code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt', transform)</code></pre>
<pre><code>testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt', transform)</code></pre>
<pre><code>train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)</code></pre>
<pre><code>test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)</code></pre>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):</code></pre>
<pre><code>    def __init__(self, pretrained):</code></pre>
<pre><code>        super(lastLayer, self).__init__()</code></pre>
<pre><code>        self.pretrained = pretrained</code></pre>
<pre><code>        self.last = nn.Sequential(</code></pre>
<pre><code>            nn.Dropout(p=0.2, inplace=True),</code></pre>
<pre><code>            nn.Linear(1408, 90),</code></pre>
<pre><code>            nn.Dropout(p=0.3, inplace=True),</code></pre>
<pre><code>            nn.Linear(90, 30),</code></pre>
<pre><code>            nn.Dropout(p=0.1, inplace=True),</code></pre>
<pre><code>            nn.Linear(30, 3)</code></pre>
<pre><code>        )</code></pre>
<pre><code>    def forward(self, x):</code></pre>
<pre><code>        x = self.pretrained(x)</code></pre>
<pre><code>        x = self.last(x)</code></pre>
<pre><code>        return x</code></pre>
</div>

<!-- Set Trainable Parameters -->
<div>
<h2>Set Trainable Parameters</h2>
<p>This part sets the parameters to be updated during training.</p>
<pre><code>params_to_update = []</code></pre>
<pre><code>for name, param in net.named_parameters():</code></pre>
<pre><code>    if param.requires_grad == True:</code></pre>
<pre><code>        params_to_update.append(param)</code></pre>
<pre><code>optimizer = torch.optim.RMSprop(params_to_update, lr=2e-4)</code></pre>
</div>

<!-- Train the Model -->
<div>
<h2>Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<pre><code>epochs = 20</code></pre>
<pre><code>from time import time</code></pre>
<pre><code>accs_train = []</code></pre>
<pre><code>loss_train = []</code></pre>
<pre><code>accs_test = []</code></pre>
<pre><code>loss_test = []</code></pre>
<pre><code>for e in range(epochs):</code></pre>
<pre><code>    start_time = time()</code></pre>
<pre><code>    accs_train, loss_train = train_epoch(net, criterion, optimizer, train_loader, accs_train, loss_train)</code></pre>
<pre><code>    accs_test, loss_test = eval_epoch(net, criterion, test_loader, accs_test, loss_test)</code></pre>
<pre><code>    if accs_test[-1] == max(accs_test):</code></pre>
<pre><code>        torch.save(net.state_dict(), 'scene_modal_en.pth')</code></pre>
<pre><code>    end_time = time()</code></pre>
<pre><code>    print(f'Epoch {e+1:3} finished in {end_time - start_time:.2f}s')</code></pre>
</div>

<!-- Plot Model Loss -->
<div>
<h2>Plot Model Loss</h2>
<p>This section plots the training and testing losses.</p>
<pre><code>import matplotlib.pyplot as plt</code></pre>
<pre><code>plt.plot(np.array(loss_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(loss_train), 'b')</code></pre>
<pre><code>plt.title('Model loss')</code></pre>
<pre><code>plt.ylabel('Loss')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.savefig('loss4.jpg')</code></pre>
<pre><code>plt.show()</code></pre>
</div>

<!-- Plot Model Accuracy -->
<div>
<h2>Plot Model Accuracy</h2>
<p>This section plots the training and testing accuracies.</p>
<pre><code>plt.plot(np.array(accs_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(accs_train), 'b')</code></pre>
<pre><code>plt.title('Model Accuracy')</code></pre>
<pre><code>plt.ylabel('Accuracy')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.savefig('acc4.jpg')</code></pre>
<pre><code>plt.show()</code></pre>
</div>

<!-- Best Accuracy -->
<div>
<h2>Best Accuracy</h2>
<p>This part displays the best accuracy achieved during training.</p>
<pre><code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code></pre>
</div>

</body>
</html>
