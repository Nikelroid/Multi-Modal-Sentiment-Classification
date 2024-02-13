<from class="colab import drive">
<drive.mount('/content/drive')>
<%cd drive/My Drive/deep_learning/Project>
<h2>Run this just for the first time</h2>
<cp train_ende.zip .>
<cp test.zip .>
<! git clone https://github.com/XL2248/MSCTD>
<cp MSCTD/MSCTD_data/ende/english_*.txt .>
<cp MSCTD/MSCTD_data/ende/image_index_*.txt .>
<cp MSCTD/MSCTD_data/ende/sentiment_*.txt .>
<!pip install --upgrade --no-cache-dir gdown>
<!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj>
<!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W>
<%%bash>
<for x in dataset/*.zip>
<do>
<  unzip -qq $x>
<done;>
<!mkdir dataset>
<!cd dataset; mkdir train test dev>
<!mv *train* dataset/train>
<!mv *test* dataset/test>
<!mv *dev* dataset/dev>
<h1>Dataset and Dataloader</h1>
<import torch>
<from torchvision import transforms as T>
<from torchvision.io import read_image>
<from torch.utils.data import Dataset>
<import torch.nn as nn>
<from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights>
<from PIL import Image>
<import os>
<from pathlib import Path>
<import pandas as pd>
<import numpy as np>
<import matplotlib.pyplot as plt>
<%matplotlib inline>
<from itertools import groupby>
<import seaborn as sns>
<from class="mtcnn.mtcnn" import MTCNN>
<!pip install git+https://github.com/elliottzheng/face-detection.git@master>
<h2>MSCTD_Dataset Class Definition</h2>
<p>A class for handling the MSCTD dataset.</p>
<p>Parameters:</p>
<ul>
  <li><code>dataset_dir</code> (str): The directory containing the dataset.</li>
  <li><code>images_dir</code> (str): The directory containing the images.</li>
  <li><code>conversation_dir</code> (str): The directory containing the conversation data.</li>
  <li><code>texts</code> (str): The path to the file containing text data.</li>
  <li><code>sentiments</code> (str): The path to the file containing sentiment data.</li>
  <li><code>transform</code> (torchvision.transforms): Optional transform to be applied to the data.</li>
</ul>
<pre><code>class MSCTD_Dataset (Dataset):
  def __init__(self, dataset_dir, images_dir, conversation_dir, texts, sentiments, transform):
    # Initialize dataset paths
    self.dataset_path = Path(dataset_dir)
    self.images_path = self.dataset_path / images_dir
    self.sentiment_path = self.dataset_path / sentiments
    self.text_path = self.dataset_path / texts
    self.conversations_path = self.dataset_path / conversation_dir
    self.transform = transform

    # Open sentiment file and get length
    with open(self.sentiment_path, 'r') as f:
      self.length = len(f.readlines())

    # Open text file and read lines
    with open(self.text_path, 'r') as f:
        self.texts = f.read().splitlines()

    # Open sentiment file and read lines as numpy array
    with open(self.sentiment_path, 'r') as f:
        self.sentiments = np.array(f.read().splitlines()).astype("int32")
    
    # Open conversation file and read lines as numpy array
    with open(self.conversations_path, 'r') as f:
        self.conversations = np.array(f.read().splitlines())
    
  def __len__(self):
        return self.length

  def __getitem__(self, idx):
        # Get image path
        img_path = self.images_path / f'{idx}.jpg'
        # Open image and convert to numpy array
        image = np.divide(np.array(Image.open(img_path)),255)

        # Apply transformation if available
        if self.transform:
            image = self.transform(image)
       
        # Get text and sentiment
        txt = self.texts[idx].strip()      
        sentiment = self.sentiments[idx]

        # Return data dictionary
        data_dict = {"text":txt,
                     "image":image,
                     "sentiment":sentiment}
        return image,sentiment
</code></pre>
<h2>Define Transform</h2>
<p>Define the transformation to be applied to the images.</p>
<pre><code>import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Resize((288,288),transforms.InterpolationMode("bicubic"))
                                ,transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
</code></pre>
<h2>Initialize Datasets</h2>
<p>Initialize the train and test datasets using the MSCTD_Dataset class.</p>
<pre><code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt',transform)
testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt',transform)
image, sentiment = testset[10]

image
</code></pre>
<h2>Set Device</h2>
<p>Set the device to GPU if available, otherwise use CPU.</p>
<pre><code>device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
</code></pre>
<h2>Import Required Modules</h2>
<p>Import the required modules for data loading and training.</p>
<pre><code>import torchvision
data_dir = './data'

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
</code></pre>
<h2>Training and Evaluation Functions</h2>
<p>Define functions for training and evaluating the model.</p>
<pre><code>from tqdm import tqdm
def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,   accs_train ,loss_train):

    epoch_loss = 0
    epoch_true = 0
    epoch_all = 0
    i = 0

    net.train()
    optimizer.zero_grad()

    with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, (x, y) in pbar: 
            x = x.to(device).float()
            y = y.to(device).to(torch.int64)
            
            p = net(x).float()
            loss = criterion(p, y)
            epoch_loss += float(loss)
            predictions = p.argmax(-1)
            epoch_all += len(predictions)
            epoch_true += (predictions == y).sum()
            pbar.set_description(f'Loss: {epoch_loss / (i + 1):.3e} - Acc: {epoch_true * 100. / epoch_all:.2f}%')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
          
        accs_train.append(float(epoch_true / epoch_all))
        loss_train.append(float(epoch_loss / (i + 1)))
    return accs_train,loss_train

def eval_epoch(net: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader,    accs_test ,loss_test ):

    epoch_loss = 0
    epoch_true = 0
    epoch_true_topfive = 0
    epoch_all = 0
    i = 0

    net.eval()
    with torch.no_grad(), tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, (x,y) in pbar:
            
            x = x.to(device).float()
            y = y.to(device).to(torch.int64)
            p = net(x).float()
            loss = criterion(p, y)
            epoch_loss += float(loss)

            # predict 
            predictions = p.argmax(-1)
            epoch_all += len(predictions)
            epoch_true += (predictions == y).sum()

            pbar.set_description(f'Loss: {epoch_loss / (i + 1):.3e} - Acc: {epoch_true * 100. / epoch_all:.2f}% ')

        accs_test.append(float(epoch_true / epoch_all))
        loss_test.append(float(epoch_loss / (i + 1)))
    return accs_test,loss_test
</code></pre>
<h2>Define Model Architecture</h2>
<p>Define the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):
    def __init__(self, pretrained):
        super(lastLayer, self).__init__()
        self.pretrained = pretrained
        self.last = nn.Sequential(
            nn.Dropout(p = 0.2,inplace=True),
            nn.Linear(1408, 90),
            nn.Dropout(p = 0.3,inplace=True),
            nn.Linear(90, 30),
            nn.Dropout(p = 0.1,inplace=True),
            nn.Linear(30, 3),
            )
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.last(x)
        return x



net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
net.classifier = nn.Sequential()

for param in net.parameters():
      param.requires_grad = False

net = lastLayer(net).to(device)
criterion = nn.CrossEntropyLoss().to(device)
</code></pre>
<h2>Set Trainable Parameters</h2>
<p>Set the parameters to be updated during training.</p>
<pre><code>print("Params to learn:")
params_to_update = []
for name,param in net.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

optimizer = torch.optim.RMSprop(params_to_update, lr=2e-4)
</code></pre>
<h2>Train the Model</h2>
<p>Train the model for a specified number of epochs.</p>
<pre><code>epochs = 20
from time import time
accs_train = []
loss_train = []
accs_test = []
loss_test = []


for e in range(epochs):
    start_time = time()
    accs_train,loss_train = train_epoch(net, criterion, optimizer, train_loader,accs_train,loss_train)
    accs_test,loss_test = eval_epoch(net, criterion, test_loader,accs_test,loss_test)
    if accs_test[-1]==max(accs_test):
      torch.save(net.state_dict(), 'scene_modal_en.pth')
    end_time = time()

    print(f'Epoch {e+1:3} finished in {end_time - start_time:.2f}s')

plt.plot(np.array(loss_test), 'r')
plt.plot(np.array(loss_train), 'b')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Train'])
plt.savefig('loss4.jpg')
plt.show()

plt.plot(np.array(accs_test), 'r')
plt.plot(np.array(accs_train), 'b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test', 'Train'])
plt.savefig('acc4.jpg')
plt.show()


print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')
</code></pre>
