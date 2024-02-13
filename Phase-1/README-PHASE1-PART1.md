
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
<pre><code>from google.colab import drive</code></pre>
<pre><code>drive.mount('/content/drive')</code></pre>
<pre><code>%cd drive/My Drive/</code></pre>
</div>

<!-- Copy Files and Clone Repository -->
<div>
<h2>Copy Files and Clone Repository</h2>
<p>This section copies necessary files and clones a GitHub repository.</p>
<pre><code>!cp train_ende.zip .</code></pre>
<pre><code>!cp test.zip .</code></pre>
<pre><code>! git clone https://github.com/XL2248/</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/english_*.txt .</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/image_index_*.txt .</code></pre>
<pre><code>!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .</code></pre>
<pre><code>!pip install --upgrade --no-cache-dir gdown</code></pre>
<pre><code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj</code></pre>
<pre><code>!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W</code></pre>
</div>

<!-- Unzip Files -->
<div>
<h2>Unzip Files</h2>
<p>This part unzips the downloaded zip files.</p>
<pre><code>%%bash</code></pre>
<pre><code>for x in *.zip</code></pre>
<pre><code>do</code></pre>
<pre><code>  unzip -qq $x</code></pre>
<pre><code>done;</code></pre>
</div>

<!-- Create Dataset Directory -->
<div>
<h2>Create Dataset Directory</h2>
<p>This section creates a directory for the dataset.</p>
<pre><code>!mkdir dataset</code></pre>
<pre><code>!cd dataset; mkdir train test dev</code></pre>
<pre><code>!mv *train* dataset/train</code></pre>
<pre><code>!mv *test* dataset/test</code></pre>
<pre><code>!mv *dev* dataset/dev</code></pre>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Dataset and Dataloader</h2>
<p>This part sets up the dataset and dataloader for training and testing the model.</p>
<pre><code>import torch</code></pre>
<pre><code>from torchvision import transforms as T</code></pre>
<pre><code>from PIL import Image</code></pre>
<pre><code>import os</code></pre>
<pre><code>from pathlib import Path</code></pre>
<pre><code>import numpy as np</code></pre>
<pre><code>from torch.utils.data import Dataset, DataLoader</code></pre>
<pre><code>import torchvision.transforms as transforms</code></pre>
<pre><code>transform = transforms.Compose([</code></pre>
<pre><code>    transforms.ToTensor(),</code></pre>
<pre><code>    transforms.Resize((288,288), interpolation=Image.BICUBIC),</code></pre>
<pre><code>    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])</code></pre>
<pre><code>])</code></pre>
</div>

<!-- Define MSCTD Dataset Class -->
<div>
<h2>Define MSCTD Dataset Class</h2>
<p>This section defines the MSCTD dataset class.</p>
<pre><code>class MSCTD_Dataset (Dataset):</code></pre>
<pre><code>  def __init__(self, dataset_dir, images_dir, conversation_dir, texts, sentiments, transform=None):</code></pre>
<pre><code>    # Constructor code...</code></pre>
<pre><code>  def __len__(self):</code></pre>
<pre><code>    # __len__ method code...</code></pre>
<pre><code>  def __getitem__(self, idx):</code></pre>
<pre><code>    # __getitem__ method code...</code></pre>
</div>

<!-- Define Face_Dataset Class -->
<div>
<h2>Define Face_Dataset Class</h2>
<p>This section defines the Face_Dataset class.</p>
<pre><code>class Face_Dataset(Dataset):</code></pre>
<pre><code>  def __init__(self, path, transform):</code></pre>
<pre><code>    # Constructor code...</code></pre>
<pre><code>  def __len__(self):</code></pre>
<pre><code>    # __len__ method code...</code></pre>
<pre><code>  def __getitem__(self, idx):</code></pre>
<pre><code>    # __getitem__ method code...</code></pre>
</div>

<!-- Train the Model -->
<div>
<h2>Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<pre><code>def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,   accs_train ,loss_train):</code></pre>
<pre><code>  # train_epoch method code...</code></pre>
<pre><code>def eval_epoch(net: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader,    accs_test ,loss_test ):</code></pre>
<pre><code>  # eval_epoch method code...</code></pre>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):</code></pre>
<pre><code>  def __init__(self, pretrained):</code></pre>
<pre><code>    # Constructor code...</code></pre>
<pre><code>  def forward(self, x):</code></pre>
<pre><code>    # forward method code...</code></pre>
</div>

<!-- Plot Model Loss -->
<div>
<h2>Plot Model Loss</h2>
<p>This section plots the training and testing losses.</p>
<pre><code>import matplotlib.pyplot as plt</code></pre>
<pre><code>plt.plot(np.array(loss_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(loss_train), 'b')</code></pre>
<pre><code>plt.title('Face Model loss')</code></pre>
<pre><code>plt.ylabel('Loss')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.show()</code></pre>
</div>

<!-- Plot Model Accuracy -->
<div>
<h2>Plot Model Accuracy</h2>
<p>This section plots the training and testing accuracies.</p>
<pre><code>plt.plot(np.array(accs_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(accs_train), 'b')</code></pre>
<pre><code>plt.title('Face Model Accuracy')</code></pre>
<pre><code>plt.ylabel('Accuracy')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.show()</code></pre>
</div>

<!-- Confusion Matrix -->
<div>
<h2>Confusion Matrix</h2>
<p>This part creates a confusion matrix to evaluate the model's performance.</p>
<pre><code>from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay</code></pre>
<pre><code>cm = confusion_matrix(truth, predict)</code></pre>
<pre><code>disp = ConfusionMatrixDisplay(cm,display_labels=['Positive','Neutral','Negative'])</code></pre>
<pre><code>fig, ax = plt.subplots(figsize=(8,8))</code></pre>
<pre><code>ax.set_title('Confusion Matrix for Face modal',fontweight="bold", size=20)</code></pre>
<pre><code>ax.set_ylabel('Pred',fontweight="bold", fontsize = 10.0)</code></pre>
<pre><code>ax.set_xlabel('True',fontweight="bold", fontsize = 10)</code></pre>
<pre><code>disp.plot(ax=ax)</code></pre>
</div>


<div>
<h2>Third step: data augmentation</h2>
<pre><code>from torch import nn</code></pre>
<pre><code>import torch</code></pre>
<pre><code>coefs = torch.rand(3)</code></pre>
<pre><code>c= nn.Softmax(dim=0)(coefs)</code></pre>
<pre><code>print(coefs)</code></pre>
<pre><code>print(c)</code></pre>
<pre><code>!pip install einops --upgrade</code></pre>
<pre><code>import numpy as np</code></pre>
<pre><code>import torch</code></pre>
<pre><code>from einops import parse_shape, rearrange</code></pre>
<pre><code>class RandomFilter(torch.nn.Module):</code></pre>
<pre><code>    # Class definition...</code></pre>
<pre><code>class RandomSmoothColor(torch.nn.Module):</code></pre>
<pre><code>    # Class definition...</code></pre>
<pre><code>class Diffeo(torch.nn.Module):</code></pre>
<pre><code>    # Class definition...</code></pre>
<pre><code>@functools.lru_cache()</code></pre>
<pre><code>def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):</code></pre>
<pre><code>    # scalar_field_modes function definition...</code></pre>
<pre><code>def scalar_field(n, m, device='cpu'):</code></pre>
<pre><code>    # scalar_field function definition...</code></pre>
<pre><code>def deform(image, T, cut, interp='linear'):</code></pre>
<pre><code>    # deform function definition...</code></pre>
<pre><code>def remap(a, dx, dy, interp):</code></pre>
<pre><code>    # remap function definition...</code></pre>
<pre><code>def temperature_range(n, cut):</code></pre>
<pre><code>    # temperature_range function definition...</code></pre>
<pre><code>COLOURCURR = RandomSmoothColor(100,0.01).to(device)</code></pre>
<pre><code>spectral = Diffeo(1., 1., 1., 1., 2, 100, 1.0).to(device)</code></pre>
<pre><code>spatial = RandomFilter(3,1).to(device)</code></pre>
<pre><code>def augment(images,tr):</code></pre>
<pre><code>    # augment function definition...</code></pre>
<pre><code>transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])</code></pre>
<pre><code>aug_face_trainset = Face_Dataset('dataset/train/aug',transform)</code></pre>
<pre><code>aug_face_testset = Face_Dataset('dataset/test/aug',transform)</code></pre>
<pre><code>aug_trainface_loader = torch.utils.data.DataLoader(aug_face_trainset, batch_size=64, shuffle=True, num_workers=2)</code></pre>
<pre><code>aug_testface_loader = torch.utils.data.DataLoader(aug_face_testset, batch_size=64, shuffle=False, num_workers=2)</code></pre>
<pre><code>def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader,   accs_train ,loss_train):</code></pre>
<pre><code>    # train_epoch function definition...</code></pre>
<pre><code>def eval_epoch(net: nn.Module, criterion: nn.Module, dataloader: torch.utils.data.DataLoader,    accs_test ,loss_test ):</code></pre>
<pre><code>    # eval_epoch function definition...</code></pre>
<pre><code>device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")</code></pre>
<pre><code>class lastLayer(nn.Module):</code></pre>
<pre><code>    # lastLayer class definition...</code></pre>
<pre><code>net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)</code></pre>
<pre><code>net.classifier = nn.Sequential()</code></pre>
<pre><code>net = lastLayer(net).to(device)</code></pre>
<pre><code>criterion = nn.CrossEntropyLoss().to(device)</code></pre>
<pre><code>params_to_update = []</code></pre>
<pre><code>for name,param in net.named_parameters():</code></pre>
<pre><code>    params_to_update.append(param)</code></pre>
<pre><code>optimizer = torch.optim.RMSprop(params_to_update, lr=5e-4)</code></pre>
<pre><code>epochs = 20</code></pre>
<pre><code>from time import time</code></pre>
<pre><code>accs_train = []</code></pre>
<pre><code>loss_train = []</code></pre>
<pre><code>accs_test = []</code></pre>
<pre><code>loss_test = []</code></pre>
<pre><code>net.load_state_dict(torch.load("models/face_modal.pth"))</code></pre>
<pre><code>for e in range(epochs):</code></pre>
<pre><code>    # Training loop...</code></pre>
<pre><code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code></pre>
<pre><code>plt.plot(np.array(loss_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(loss_train), 'b')</code></pre>
<pre><code>plt.title('Face Model loss')</code></pre>
<pre><code>plt.ylabel('Loss')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.show()</code></pre>
<pre><code>plt.plot(np.array(accs_test), 'r')</code></pre>
<pre><code>plt.plot(np.array(accs_train), 'b')</code></pre>
<pre><code>plt.title('Face Model Accuracy')</code></pre>
<pre><code>plt.ylabel('Accuracy')</code></pre>
<pre><code>plt.xlabel('Epoch')</code></pre>
<pre><code>plt.legend(['Test', 'Train'])</code></pre>
<pre><code>plt.show()</code></pre>
</div>
</body>
</html>
