<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>README</title>
</head>
<body>
<!-- Mount Google Drive -->
<div>
<h2>Mount Google Drive</h2>
<p>This code mounts Google Drive to the Colab environment. It's necessary for accessing files stored in Google Drive.</p>
<code>from google.colab import drive</code><br>
<code>drive.mount('/content/drive')</code>
</div>

<!-- Change Directory -->
<div>
<h2>Change Directory</h2>
<p>This section changes the directory to the project directory and performs initial setup tasks like copying files, cloning repositories, and unzipping datasets.</p>
<code>%cd drive/My Drive/deep_learning/Project</code><br>
<em>Run this just for the first time:</em><br>
<code>!cp train_ende.zip .</code><br>
<code>!cp test.zip .</code><br>
<code>!git clone https://github.com/XL2248/MSCTD</code><br>
<code>!cp MSCTD/MSCTD_data/ende/english_*.txt .</code><br>
<code>!cp MSCTD/MSCTD_data/ende/image_index_*.txt .</code><br>
<code>!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .</code><br>
<code>!pip install --upgrade --no-cache-dir gdown</code><br>
<code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj</code><br>
<code>!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W</code><br>
<code>%%bash</code><br>
<code>for x in dataset/*.zip</code><br>
<code>do</code><br>
<code>  unzip -qq $x</code><br>
<code>done;</code><br>
<code>!mkdir dataset</code><br>
<code>!cd dataset; mkdir train test dev</code><br>
<code>!mv *train* dataset/train</code><br>
<code>!mv *test* dataset/test</code><br>
<code>!mv *dev* dataset/dev</code>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Dataset and Dataloader</h2>
<p>This part involves setting up the dataset and dataloader for training and testing the model.</p>
<code>import torch</code><br>
<code>from torchvision import transforms as T</code><br>
<code>from PIL import Image</code><br>
<code>import os</code><br>
<code>from pathlib import Path</code><br>
<code>import numpy as np</code><br>
<code>from torch.utils.data import Dataset, DataLoader</code><br>
<code>import torchvision.transforms as transforms</code><br>
<code>transform = transforms.Compose([</code><br>
<code>    transforms.ToTensor(),</code><br>
<code>    transforms.Resize((288,288), interpolation=Image.BICUBIC),</code><br>
<code>    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</code><br>
<code>])</code><br>
<code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt', transform)</code><br>
<code>testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt', transform)</code><br>
<code>train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)</code><br>
<code>test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)</code>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<code>class lastLayer(nn.Module):</code><br>
<code>    def __init__(self, pretrained):</code><br>
<code>        super(lastLayer, self).__init__()</code><br>
<code>        self.pretrained = pretrained</code><br>
<code>        self.last = nn.Sequential(</code><br>
<code>            nn.Dropout(p=0.2, inplace=True),</code><br>
<code>            nn.Linear(1408, 90),</code><br>
<code>            nn.Dropout(p=0.3, inplace=True),</code><br>
<code>            nn.Linear(90, 30),</code><br>
<code>            nn.Dropout(p=0.1, inplace=True),</code><br>
<code>            nn.Linear(30, 3)</code><br>
<code>        )</code><br>
<code>    def forward(self, x):</code><br>
<code>        x = self.pretrained(x)</code><br>
<code>        x = self.last(x)</code><br>
<code>        return x</code>
</div>

<!-- Set Trainable Parameters -->
<div>
<h2>Set Trainable Parameters</h2>
<p>This part sets the parameters to be updated during training.</p>
<code>params_to_update = []</code><br>
<code>for name, param in net.named_parameters():</code><br>
<code>    if param.requires_grad == True:</code><br>
<code>        params_to_update.append(param)</code><br>
<code>optimizer = torch.optim.RMSprop(params_to_update, lr=2e-4)</code>
</div>

<!-- Train the Model -->
<div>
<h2>Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<code>epochs = 20</code><br>
<code>from time import time</code><br>
<code>accs_train = []</code><br>
<code>loss_train = []</code><br>
<code>accs_test = []</code><br>
<code>loss_test = []</code><br>
<code>for e in range(epochs):</code><br>
<code>    start_time = time()</code><br>
<code>    accs_train, loss_train = train_epoch(net, criterion, optimizer, train_loader, accs_train, loss_train)</code><br>
<code>    accs_test, loss_test = eval_epoch(net, criterion, test_loader, accs_test, loss_test)</code><br>
<code>    if accs_test[-1] == max(accs_test):</code><br>
<code>        torch.save(net.state_dict(), 'scene_modal_en.pth')</code><br>
<code>    end_time = time()</code><br>
<code>    print(f'Epoch {e+1:3} finished in {end_time - start_time:.2f}s')</code><br>
<code>import matplotlib.pyplot as plt</code><br>
<code>plt.plot(np.array(loss_test), 'r')</code><br>
<code>plt.plot(np.array(loss_train), 'b')</code><br>
<code>plt.title('Model loss')</code><br>
<code>plt.ylabel('Loss')</code><br>
<code>plt.xlabel('Epoch')</code><br>
<code>plt.legend(['Test', 'Train'])</code><br>
<code>plt.savefig('loss4.jpg')</code><br>
<code>plt.show()</code><br>
<code>plt.plot(np.array(accs_test), 'r')</code><br>
<code>plt.plot(np.array(accs_train), 'b')</code><br>
<code>plt.title('Model Accuracy')</code><br>
<code>plt.ylabel('Accuracy')</code><br>
<code>plt.xlabel('Epoch')</code><br>
<code>plt.legend(['Test', 'Train'])</code><br>
<code>plt.savefig('acc4.jpg')</code><br>
<code>plt.show()</code><br>
</div>

<div>
<h2>Best Accuracy</h2>
<p>This part displays the best accuracy achieved during training.</p>
<code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code>
</div>

</body>
</html>
