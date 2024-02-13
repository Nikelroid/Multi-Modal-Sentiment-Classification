
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
<pre><code>from google.colab import drive</code>
<code>drive.mount('/content/drive')</code>
<code>%cd drive/My Drive/</code></pre>
</div>

<!-- Copy Files and Clone Repository -->
<div>
<h2>Copy Files and Clone Repository</h2>
<p>This section copies necessary files and clones a GitHub repository.</p>
<pre><code>!cp train_ende.zip .</code>
<code>!cp test.zip .</code>
<code>! git clone https://github.com/XL2248/</code>
<code>!cp MSCTD/MSCTD_data/ende/english_*.txt .</code>
<code>!cp MSCTD/MSCTD_data/ende/image_index_*.txt .</code>
<code>!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .</code>
<code>!pip install --upgrade --no-cache-dir gdown</code>
<code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj</code>
<code>!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W</code></pre>
</div>

<!-- Unzip Files -->
<div>
<h2>Unzip Files</h2>
<p>This part unzips the downloaded zip files.</p>
<pre><code>%%bash</code>
<code>for x in *.zip</code>
<code>do</code>
<code>  unzip -qq $x</code>
<code>done;</code></pre>
</div>

<!-- Create Dataset Directory -->
<div>
<h2>Create Dataset Directory</h2>
<p>This section creates a directory for the dataset.</p>
<pre><code>!mkdir dataset</code>
<code>!cd dataset; mkdir train test dev</code>
<code>!mv *train* dataset/train</code>
<code>!mv *test* dataset/test</code>
<code>!mv *dev* dataset/dev</code></pre>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Dataset and Dataloader</h2>
<p>This part sets up the dataset and dataloader for training and testing the model.</p>
<pre><code>import torch</code>
<code>from torchvision import transforms as T</code>
<code>from PIL import Image</code>
<code>import os</code>
<code>from pathlib import Path</code>
<code>import numpy as np</code>
<code>from torch.utils.data import Dataset, DataLoader</code>
<code>import torchvision.transforms as transforms</code>
<code>transform = transforms.Compose([</code>
<code>    transforms.ToTensor(),</code>
<code>    transforms.Resize((288,288), interpolation=Image.BICUBIC),</code>
<code>    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</code>
<code>])</code></pre>
</div>

<!-- Define MSCTD Dataset Class -->
<div>
<h2>Define MSCTD Dataset Class</h2>
<p>This section defines the MSCTD dataset class.</p>
<pre><code>class MSCTD_Dataset (Dataset):</code>
<code>  def __init__(self, dataset_dir, images_dir, conversation_dir, texts, sentiments, transform=None):</code>
<code>    # Constructor code...</code>
<code>  def __len__(self):</code>
<code>    # __len__ method code...</code>
<code>  def __getitem__(self, idx):</code>
<code>    # __getitem__ method code...</code></pre>
</div>

<!-- Define Face_Dataset Class -->
<div>
<h2>Define Face_Dataset Class</h2>
<p>This section defines the Face_Dataset class.</p>
<pre><code>class Face_Dataset(Dataset):</code>
<code>  def __init__(self, path, transform):</code>
<code>    # Constructor code...</code>
<code>  def __len__(self):</code>
<code>    # __len__ method code...</code>
<code>  def __getitem__(self, idx):</code>
<code>    # __getitem__ method code...</code></pre>
</div>

<!-- Train the Model -->
<div>
<h2>Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<pre><code>def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,   accs_train ,loss_train):</code>
<code>  # train_epoch method code...</code>
<code>def eval_epoch(net: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader,    accs_test ,loss_test ):</code>
<code>  # eval_epoch method code...</code></pre>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):</code>
<code>  def __init__(self, pretrained):</code>
<code>    # Constructor code...</code>
<code>  def forward(self, x):</code>
<code>    # forward method code...</code></pre>
</div>

<!-- Plot Model Loss -->
<div>
<h2>Plot Model Loss</h2>
<p>This section plots the training and testing losses.</p>
<pre><code>import matplotlib.pyplot as plt</code>
<code>plt.plot(np.array(loss_test), 'r')</code>
<code>plt.plot(np.array(loss_train), 'b')</code>
<code>plt.title('Face Model loss')</code>
<code>plt.ylabel('Loss')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.show()</code></pre>
</div>

<!-- Plot Model Accuracy -->
<div>
<h2>Plot Model Accuracy</h2>
<p>This section plots the training and testing accuracies.</p>
<pre><code>plt.plot(np.array(accs_test), 'r')</code>
<code>plt.plot(np.array(accs_train), 'b')</code>
<code>plt.title('Face Model Accuracy')</code>
<code>plt.ylabel('Accuracy')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.show()</code></pre>
</div>

<!-- Confusion Matrix -->
<div>
<h2>Confusion Matrix</h2>
<p>This part creates a confusion matrix to evaluate the model's performance.</p>
<pre><code>from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay</code>
<code>cm = confusion_matrix(truth, predict)</code>
<code>disp = ConfusionMatrixDisplay(cm,display_labels=['Positive','Neutral','Negative'])</code>
<code>fig, ax = plt.subplots(figsize=(8,8))</code>
<code>ax.set_title('Confusion Matrix for Face modal',fontweight="bold", size=20)</code>
<code>ax.set_ylabel('Pred',fontweight="bold", fontsize = 10.0)</code>
<code>ax.set_xlabel('True',fontweight="bold", fontsize = 10)</code>
<code>disp.plot(ax=ax)</code></pre>
</div>


<div>
<h2>Third step: data augmentation</h2>
<pre><code>from torch import nn</code>
<code>import torch</code>
<code>coefs = torch.rand(3)</code>
<code>c= nn.Softmax(dim=0)(coefs)</code>
<code>print(coefs)</code>
<code>print(c)</code>
<code>!pip install einops --upgrade</code>
<code>import numpy as np</code>
<code>import torch</code>
<code>from einops import parse_shape, rearrange</code>
<code>class RandomFilter(torch.nn.Module):</code>
<code>    # Class definition...</code>
<code>class RandomSmoothColor(torch.nn.Module):</code>
<code>    # Class definition...</code>
<code>class Diffeo(torch.nn.Module):</code>
<code>    # Class definition...</code>
<code>@functools.lru_cache()</code>
<code>def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):</code>
<code>    # scalar_field_modes function definition...</code>
<code>def scalar_field(n, m, device='cpu'):</code>
<code>    # scalar_field function definition...</code>
<code>def deform(image, T, cut, interp='linear'):</code>
<code>    # deform function definition...</code>
<code>def remap(a, dx, dy, interp):</code>
<code>    # remap function definition...</code>
<code>def temperature_range(n, cut):</code>
<code>    # temperature_range function definition...</code>
<code>COLOURCURR = RandomSmoothColor(100,0.01).to(device)</code>
<code>spectral = Diffeo(1., 1., 1., 1., 2, 100, 1.0).to(device)</code>
<code>spatial = RandomFilter(3,1).to(device)</code>
<code>def augment(images,tr):</code>
<code>    # augment function definition...</code>
<code>transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])</code>
<code>aug_face_trainset = Face_Dataset('dataset/train/aug',transform)</code>
<code>aug_face_testset = Face_Dataset('dataset/test/aug',transform)</code>
<code>aug_trainface_loader = torch.utils.data.DataLoader(aug_face_trainset, batch_size=64, shuffle=True, num_workers=2)</code>
<code>aug_testface_loader = torch.utils.data.DataLoader(aug_face_testset, batch_size=64, shuffle=False, num_workers=2)</code>
<code>def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,   accs_train ,loss_train):</code>
<code>    # train_epoch function definition...</code>
<code>def eval_epoch(net: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader,    accs_test ,loss_test ):</code>
<code>    # eval_epoch function definition...</code>
<code>device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")</code>
<code>class lastLayer(nn.Module):</code>
<code>    # lastLayer class definition...</code>
<code>net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)</code>
<code>net.classifier = nn.Sequential()</code>
<code>net = lastLayer(net).to(device)</code>
<code>criterion = nn.CrossEntropyLoss().to(device)</code>
<code>params_to_update = []</code>
<code>for name,param in net.named_parameters():</code>
<code>    params_to_update.append(param)</code>
<code>optimizer = torch.optim.RMSprop(params_to_update, lr=5e-4)</code>
<code>epochs = 20</code>
<code>from time import time</code>
<code>accs_train = []</code>
<code>loss_train = []</code>
<code>accs_test = []</code>
<code>loss_test = []</code>
<code>net.load_state_dict(torch.load("models/face_modal.pth"))</code>
<code>for e in range(epochs):</code>
<code>    # Training loop...</code>
<code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code>
<code>plt.plot(np.array(loss_test), 'r')</code>
<code>plt.plot(np.array(loss_train), 'b')</code>
<code>plt.title('Face Model loss')</code>
<code>plt.ylabel('Loss')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.show()</code>
<code>plt.plot(np.array(accs_test), 'r')</code>
<code>plt.plot(np.array(accs_train), 'b')</code>
<code>plt.title('Face Model Accuracy')</code>
<code>plt.ylabel('Accuracy')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.show()</code></pre>
</div>
</body>
</html>
