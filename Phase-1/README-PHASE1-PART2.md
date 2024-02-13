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
<pre><code>!cp train_ende.zip .</code>
!cp test.zip .
!git clone https://github.com/XL2248/MSCTD
!cp MSCTD/MSCTD_data/ende/english_*.txt .
!cp MSCTD/MSCTD_data/ende/image_index_*.txt .
!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .
!pip install --upgrade --no-cache-dir gdown
!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj
!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W
%%bash
for x in dataset/*.zip
do
  unzip -qq $x
done;
!mkdir dataset
!cd dataset; mkdir train test dev
!mv *train* dataset/train
!mv *test* dataset/test
!mv *dev* dataset/dev
</code></pre>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Dataset and Dataloader</h2>
<p>This part involves setting up the dataset and dataloader for training and testing the model.</p>
<pre><code>import torch</code>
<pre><code>from torchvision import transforms as T</code>
<pre><code>from PIL import Image</code>
<pre><code>import os</code>
<pre><code>from pathlib import Path</code>
<pre><code>import numpy as np</code>
<pre><code>from torch.utils.data import Dataset, DataLoader</code>
<pre><code>import torchvision.transforms as transforms</code>
<pre><code>transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((288,288), interpolation=Image.BICUBIC),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])</code>
<pre><code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt', transform)</code>
<pre><code>testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt', transform)</code>
<pre><code>train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)</code>
<pre><code>test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)</code>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):</code>
<pre><code>    def __init__(self, pretrained):</code>
<pre><code>        super(lastLayer, self).__init__()</code>
<pre><code>        self.pretrained = pretrained</code>
<pre><code>        self.last = nn.Sequential(</code>
<pre><code>            nn.Dropout(p=0.2, inplace=True),</code>
<pre><code>            nn.Linear(1408, 90),</code>
<pre><code>            nn.Dropout(p=0.3, inplace=True),</code>
<pre><code>            nn.Linear(90, 30),</code>
<pre><code>            nn.Dropout(p=0.1, inplace=True),</code>
<pre><code>            nn.Linear(30, 3)</code>
<pre><code>        )</code>
<pre><code>    def forward(self, x):</code>
<pre><code>        x = self.pretrained(x)</code>
<pre><code>        x = self.last(x)</code>
<pre><code>        return x</code>
</div>

<!-- Set Trainable Parameters -->
<div>
<h2>Set Trainable Parameters</h2>
<p>This part sets the parameters to be updated during training.</p>
<pre><code>params_to_update = []</code>
<pre><code>for name, param in net.named_parameters():</code>
<pre><code>    if param.requires_grad == True:</code>
<pre><code>        params_to_update.append(param)</code>
<pre><code>optimizer = torch.optim.RMSprop(params_to_update, lr=2e-4)</code>
</div>

<!-- Train the Model -->
<div>
<h2>Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<pre><code>epochs = 20</code>
<pre><code>from time import time</code>
<pre><code>accs_train = []</code>
<pre><code>loss_train = []</code>
<pre><code>accs_test = []</code>
<pre><code>loss_test = []</code>
<pre><code>for e in range(epochs):</code>
<pre><code>    start_time = time()</code>
<pre><code>    accs_train, loss_train = train_epoch(net, criterion, optimizer, train_loader, accs_train, loss_train)</code>
<pre><code>    accs_test, loss_test = eval_epoch(net, criterion, test_loader, accs_test, loss_test)</code>
<pre><code>    if accs_test[-1] == max(accs_test):</code>
<pre><code>        torch.save(net.state_dict(), 'scene_modal_en.pth')</code>
<pre><code>    end_time = time()</code>
<pre><code>    print(f'Epoch {e+1:3} finished in {end_time - start_time:.2f}s')</code>
</div>

<!-- Plot Model Metrics -->
<div>
<h2>Plot Model Metrics</h2>
<p>This part plots the model loss and accuracy.</p>
<pre><code>import matplotlib.pyplot as plt</code>
<pre><code>plt.plot(np.array(loss_test), 'r')</code>
<pre><code>plt.plot(np.array(loss_train), 'b')</code>
<pre><code>plt.title('Model loss')</code>
<pre><code>plt.ylabel('Loss')</code>
<pre><code>plt.xlabel('Epoch')</code>
<pre><code>plt.legend(['Test', 'Train'])</code>
<pre><code>plt.savefig('loss4.jpg')</code>
<pre><code>plt.show()</code>
<pre><code>plt.plot(np.array(accs_test), 'r')</code>
<pre><code>plt.plot(np.array(accs_train), 'b')</code>
<pre><code>plt.title('Model Accuracy')</code>
<pre><code>plt.ylabel('Accuracy')</code>
<pre><code>plt.xlabel('Epoch')</code>
<pre><code>plt.legend(['Test', 'Train'])</code>
<pre><code>plt.savefig('acc4.jpg')</code>
<pre><code>plt.show()</code>
</div>

<!-- Best Accuracy -->
<div>
<h2>Best Accuracy</h2>
<p>This part displays the best accuracy achieved during training.</p>
<pre><code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code>
</div>

</body>
</html>
