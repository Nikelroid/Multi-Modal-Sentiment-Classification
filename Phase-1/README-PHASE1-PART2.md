<div>
<h2>Phase 1 - Phase 2 : Facial Expression Prediction Model Training Pipeline</h2>
<p>This Python script orchestrates a comprehensive pipeline for training a facial expression prediction model using the Multi-Scene Character Text Dataset (MSCTD). Initially, the script sets up the environment by mounting Google Drive and navigating to the project directory. It then proceeds to download and organize the dataset, consisting of images and corresponding textual descriptions, essential for training the model. Leveraging the PyTorch library, the script defines a custom dataset class, MSCTD_Dataset, tailored to handle the dataset's structure and facilitate efficient data loading. Augmentation techniques such as resizing, normalization, and data shuffling are applied to enhance the dataset's diversity and improve model generalization. The pipeline encompasses model training and evaluation stages, employing a pre-trained EfficientNet-B2 architecture fine-tuned for facial expression prediction. Throughout training, performance metrics including accuracy and loss are monitored and visualized, enabling insightful analysis of the model's learning dynamics. Finally, the script saves the trained model weights for future use, ensuring reproducibility and scalability of the facial expression prediction solution.</p>
</div>


<div>
<h2>Part 0: Mount Google Drive</h2>
<p>This code mounts Google Drive to the Colab environment. It's necessary for accessing files stored in Google Drive.</p>
<pre><code>from google.colab import drive</code></pre>
<pre><code>drive.mount('/content/drive')</code></pre>
</div>

<!-- Change Directory -->
<div>
<h2>Part 1: Change Directory</h2>
<p>This section changes the directory to the project directory and performs initial setup tasks like copying files, cloning repositories, and unzipping datasets.</p>
<pre><code>%cd drive/My Drive/deep_learning/Project</code>
<em>Run this just for the first time:</em>
<code>!cp train_ende.zip .</code>
<code>!cp test.zip .</code>
<code>!git clone https://github.com/XL2248/MSCTD</code>
<code>!cp MSCTD/MSCTD_data/ende/english_*.txt .</code>
<code>!cp MSCTD/MSCTD_data/ende/image_index_*.txt .</code>
<code>!cp MSCTD/MSCTD_data/ende/sentiment_*.txt .</code>
<code>!pip install --upgrade --no-cache-dir gdown</code>
<code>!gdown --id 1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj</code>
<code>!gdown --id 1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W</code>
<code>%%bash</code>
<code>for x in dataset/*.zip</code>
<code>do</code>
<code>  unzip -qq $x</code>
<code>done</code>
<code>!mkdir dataset</code>
<code>!cd dataset; mkdir train test dev</code>
<code>!mv *train* dataset/train</code>
<code>!mv *test* dataset/test</code>
<code>!mv *dev* dataset/dev</code></pre>
</div>

<!-- Dataset and Dataloader -->
<div>
<h2>Part 2: Dataset and Dataloader</h2>
<p>This part involves setting up the dataset and dataloader for training and testing the model.</p>
<pre><code>import torch</code>
<code>from torchvision import transforms as T</code>
<code>from PIL import Image</code>
<code>import os</code>
<code>from pathlib import Path</code>
<code>import numpy as np</code>
<code>from torch.utils.data import Dataset, DataLoader</code>
<code>import torchvision.transforms as transforms</code>
<code>transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((288,288), interpolation=Image.BICUBIC),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])</code>
<code>trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt', transform)</code>
<code>testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt', transform)</code>
<code>train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)</code>
<code>test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)</code></pre>
</div>

<!-- Define Model Architecture -->
<div>
<h2>Part 3: Define Model Architecture</h2>
<p>This section defines the architecture of the last layer of the model.</p>
<pre><code>class lastLayer(nn.Module):</code>
<code>    def __init__(self, pretrained):</code>
<code>        super(lastLayer, self).__init__()</code>
<code>        self.pretrained = pretrained</code>
<code>        self.last = nn.Sequential(</code>
<code>            nn.Dropout(p=0.2, inplace=True),</code>
<code>            nn.Linear(1408, 90),</code>
<code>            nn.Dropout(p=0.3, inplace=True),</code>
<code>            nn.Linear(90, 30),</code>
<code>            nn.Dropout(p=0.1, inplace=True),</code>
<code>            nn.Linear(30, 3)</code>
<code>        )</code>
<code>    def forward(self, x):</code>
<code>        x = self.pretrained(x)</code>
<code>        x = self.last(x)</code>
<code>        return x</code></pre>
</div>

<!-- Set Trainable Parameters -->
<div>
<h2>Part 4: Set Trainable Parameters</h2>
<p>This part sets the parameters to be updated during training.</p>
<pre><code>params_to_update = []</code>
<code>for name, param in net.named_parameters():</code>
<code>    if param.requires_grad == True:</code>
<code>        params_to_update.append(param)</code>
<code>optimizer = torch.optim.RMSprop(params_to_update, lr=2e-4)</code></pre>
</div>

<!-- Train the Model -->
<div>
<h2>Part 5: Train the Model</h2>
<p>This section trains the model for a specified number of epochs.</p>
<pre><code>epochs = 20</code>
<code>from time import time</code>
<code>accs_train = []</code>
<code>loss_train = []</code>
<code>accs_test = []</code>
<code>loss_test = []</code>
<code>for e in range(epochs):</code>
<code>    start_time = time()</code>
<code>    accs_train, loss_train = train_epoch(net, criterion, optimizer, train_loader, accs_train, loss_train)</code>
<code>    accs_test, loss_test = eval_epoch(net, criterion, test_loader, accs_test, loss_test)</code>
<code>    if accs_test[-1] == max(accs_test):</code>
<code>        torch.save(net.state_dict(), 'scene_modal_en.pth')</code>
<code>    end_time = time()</code>
<code>    print(f'Epoch {e+1:3} finished in {end_time - start_time:.2f}s')</code></pre>
</div>

<!-- Plot Model Loss -->
<div>
<h2>Part 6: Plot Model Loss</h2>
<p>This section plots the training and testing losses.</p>
<pre><code>import matplotlib.pyplot as plt</code>
<code>plt.plot(np.array(loss_test), 'r')</code>
<code>plt.plot(np.array(loss_train), 'b')</code>
<code>plt.title('Model loss')</code>
<code>plt.ylabel('Loss')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.savefig('loss4.jpg')</code>
<code>plt.show()</code></pre>
</div>

<!-- Plot Model Accuracy -->
<div>
<h2>Part 7: Plot Model Accuracy</h2>
<p>This section plots the training and testing accuracies.</p>
<pre><code>plt.plot(np.array(accs_test), 'r')</code>
<code>plt.plot(np.array(accs_train), 'b')</code>
<code>plt.title('Model Accuracy')</code>
<code>plt.ylabel('Accuracy')</code>
<code>plt.xlabel('Epoch')</code>
<code>plt.legend(['Test', 'Train'])</code>
<code>plt.savefig('acc4.jpg')</code>
<code>plt.show()</code></pre>
</div>

<!-- Best Accuracy -->
<div>
<h2>Part 8: Best Accuracy</h2>
<p>This part displays the best accuracy achieved during training.</p>
<pre><code>print(f'Best Accuracy :{max(accs_test) * 100.:.2f}%')</code></pre>
</div>

</body>
</html>
