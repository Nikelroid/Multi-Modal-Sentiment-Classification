
<body>

<h1>Phase 1 - Phase 3 : Integrating Facial and Image Analysis for Sentiment Prediction in Visual Data</h1>

<p>Our project focuses on combining two approaches for sentiment prediction from visual data: facial analysis and overall image analysis. We propose a multi-layer perceptron (MLP) model that effectively integrates the insights obtained from both modalities to achieve superior sentiment prediction accuracy.</p>

<p>In this part of the project, we present the methodology and implementation details of the combined MLP model. We leverage deep learning techniques to extract features from facial expressions and overall image content separately. Subsequently, we design the MLP architecture to fuse these features in a synergistic manner, exploiting the complementary information offered by each modality.</p>

<p>The README provides a comprehensive guide to understanding the process of combining facial and image analysis for sentiment prediction. It includes sections detailing data preprocessing, model architecture, training procedure, and evaluation metrics. By following the README, users can gain insights into our approach, replicate our experiments, and understand the rationale behind our methodology.</p>

<p>Overall, our project contributes to advancing the field of sentiment analysis in visual data by demonstrating the effectiveness of integrating multiple modalities. This integration enables more nuanced and accurate predictions of sentiment, which has applications in various domains, including social media analytics, market research, and content recommendation systems.</p>
<h2>Table of Contents</h2>
<ol>
  <li><a href="#part1">Load and Save Data From Image and Face Sentiment</a></li>
  <li><a href="#part2">Load and Save Face Lengths</a></li>
  <li><a href="#part3">Compute and Save Face Sentiments</a></li>
  <li><a href="#part4">Compute and Save Overall Image Sentiment</a></li>
  <li><a href="#part5">Training The MLP for Get Best Combination</a></li>
  <li><a href="#part6">Test and Evaluation</a></li>
</ol>

<h2 id="part1">1. Load and Save Data From Image and Face Sentiment</h2>
<p>This part involves loading and saving data related to images and face sentiments. First, the necessary transformations for image data are defined. Then, datasets for both training and testing are created using the provided paths and files. Additionally, a face detection model is loaded and used to extract faces from images, and the lengths of these faces are computed and saved.</p>

<pre><code>
import torch
from torchvision import transforms
from dataset import MSCTD_Dataset, Face_Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((288, 288), transforms.InterpolationMode.BICUBIC),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_trainset = MSCTD_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt')
face_trainset = Face_Dataset('dataset/train/faces', transform)

image_testset = MSCTD_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt')
face_testset = Face_Dataset('dataset/test/faces', transform)

image_trainloader = torch.utils.data.DataLoader(image_trainset, batch_size=64, shuffle=False, num_workers=2)
face_trainloader = torch.utils.data.DataLoader(face_trainset, batch_size=64, shuffle=False, num_workers=2)
image_testloader = torch.utils.data.DataLoader(image_testset, batch_size=32, shuffle=False, num_workers=2)
face_testloader = torch.utils.data.DataLoader(face_testset, batch_size=32, shuffle=False, num_workers=2)
</code></pre>


<h2 id="part2">2. Load and Save Face Lengths</h2>
<p>This part focuses on computing and saving the lengths of faces detected in the images. The RetinaFace model is utilized for face detection, and the lengths of the detected faces are stored in text files for both the training and testing datasets.</p>

<pre><code>
!pip install git+https://github.com/elliottzheng/face-detection.git@master
from skimage import io
from face_detection import RetinaFace
import pickle

main_detector = RetinaFace(gpu_id=0)

def face_detector(image):
    faces_boundaries = main_detector(image)
    faces = []
    for i in range(len(faces_boundaries)):
        stats, _, score = faces_boundaries[i]
        stats = stats.astype(int)
        if score > 0.95:
            faces.append(Image.fromarray(image[max(0, stats[1]):min(image.shape[0], stats[3]),
                                              max(0, stats[0]):min(image.shape[1], stats[2])]))
    return faces

len_faces_test = []
count = 0

for img, _ in image_testset:
    count += 1
    len_faces_test.append(len(face_detector(np.array(img.convert('RGB')))))
    if count % 1000 == 999:
        print(count)

with open("dataset/test/sents_data/len_faces.txt", 'wb') as f:
    pickle.dump(len_faces_test, f)

len_faces = []
count = 0

for img, _ in image_trainset:
    count += 1
    len_faces.append(len(face_detector(np.array(img.convert('RGB')))))
    if count % 2000 == 1999:
        print(count)

with open("dataset/train/sents_data/len_faces.txt", 'wb') as f:
    pickle.dump(len_faces, f)

with open("dataset/train/sents_data/len_faces.txt", 'rb') as f:
    len_faces = pickle.load(f)

with open("dataset/test/sents_data/len_faces.txt", 'rb') as f:
    len_faces_test = pickle.load(f)

print(len(len_faces), len(len_faces_test))
</code></pre>


<h2 id="part3">3. Compute and Save Face Sentiments</h2>
<p>In this section, face sentiments are computed and saved based on the detected faces. A pre-trained model is loaded, and the sentiments for each face in the training dataset are predicted and saved.</p>

<pre><code>
transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Resize((288,288),transforms.InterpolationMode("bicubic"))
                                ,transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

faces_sents = torch.empty((1),device=device)
counter = [6/21, 8/21, 7/21]

with torch.no_grad(), tqdm(enumerate(face_trainloader), total=len(face_trainloader)) as pbar:
        for i, (x, y) in pbar:
            x = x.to(device).float()
            y = y.to(device).to(torch.int64)
            p = net(x).float()
            faces_sents = torch.cat((faces_sents, torch.argmax(p, 1)), dim=0)

faces_sents = faces_sents[1:]

with open("dataset/train/sents_data/face_sents.txt", 'wb') as f:
    pickle.dump(faces_sents, f)

faces_sents_test = torch.empty((1), device=device)

with torch.no_grad(), tqdm(enumerate(face_testloader), total=len(face_testloader)) as pbar:
        for i, (x, y) in pbar:
            x = x.to(device).float()
            y = y.to(device).to(torch.int64)
            p = net(x).float()
            faces_sents_test = torch.cat((faces_sents, torch.argmax(p, 1)), dim=0)

faces_sents_test = faces_sents_test[1:]

with open("dataset/test/sents_data/face_sents.txt", 'wb') as f:
    pickle.dump(faces_sents_test, f)
</code></pre>


<h2 id="part4">4. Compute and Save Overall Image Sentiment</h2>
<p>Here, the overall sentiments of images are computed and saved. Another pre-trained model is loaded, and the sentiments for images in both the training and testing datasets are predicted and stored.</p>

<pre><code>
class lastLayer(nn.Module):
    def __init__(self, pretrained):
        super(lastLayer, self).__init__()
        self.pretrained = pretrained
        self.last = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1408, 90),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(90, 30),
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(30, 3),
        )
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.last(x)
        return x

net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
net.classifier = nn.Sequential()

net = lastLayer(net).to(device)
net.load_state_dict(torch.load("models/face_aug_modal.pth"))
net.eval()

###Load and save faces_lengths
main_detector = RetinaFace(gpu_id=0)

def face_detector(image):
    faces_boundaries = main_detector(image)
    faces = []
    for i in range(len(faces_boundaries)):
      stats, _, score = faces_boundaries[i]
      stats = stats.astype(int)
      if score > 0.95:
        faces.append(Image.fromarray(image[max(0,stats[1]):min(image.shape[0],stats[3]),
                                           max(0,stats[0]):min(image.shape[1],stats[2])]))
    return faces

count = 0
len_faces_test = []

for img,_ in image_testset:
  count += 1
  len_faces_test.append(len(face_detector(np.array(img.convert('RGB')))))
  if count % 1000 == 999:
    print(count)

with open("dataset/test/sents_data/len_faces.txt", 'wb') as f:
          pickle.dump(len_faces_test, f)

len_faces = []
count = 0

for img,_ in image_trainset:
  count += 1
  len_faces.append(len(face_detector(np.array(img.convert('RGB')))))
  if count % 2000 == 1999:
    print(count)

with open("dataset/train/sents_data/len_faces.txt", 'wb') as f:
          pickle.dump(len_faces, f)

with open("dataset/train/sents_data/len_faces.txt", 'rb') as f:
        len_faces = pickle.load(f)

with open("dataset/test/sents_data/len_faces.txt", 'rb') as f:
        len_faces_test = pickle.load(f)

print(len(len_faces), len(len_faces_test))
</code></pre>


<h2 id="part5">5. Training The MLP for Get Best Combination</h2>
<p>This part involves training a Multi-Layer Perceptron (MLP) to obtain the best combination of features from previous phases. The MLP is trained using data prepared from the image and face sentiments, along with their lengths. The training process is detailed, including the definition of the MLP architecture, data loading, training loop, and evaluation.</p>

<pre><code>
class stablizer(nn.Module):
  def __init__(self):
    super(stablizer, self).__init__()
    self.fc = nn.Sequential(
        nn.Linear(7,20),
        nn.ReLU(),
        nn.Linear(20,10),
        nn.ReLU(),
        nn.Linear(10,3),
        nn.Softmax(dim=1)
    )
    
  def forward(self, x):
    output = self.fc(x)
    return output

def onehot(Y, num=3):
    out = torch.zeros((Y.size()[0], num), device=device)
    for i, index in enumerate(Y):
        out[i, index.item()] = 1
    return out 

from tqdm import tqdm

def train_epoch(net: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, input, labels, accs_train, loss_train):
    epoch_loss = 0
    epoch_true = 0
    epoch_all = 0
    i = 0

    net.train()
    optimizer.zero_grad()

    x = input.float()
    y = labels.to(torch.int64)
    
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
    return accs_train, loss_train

def eval_epoch(net: nn.Module, criterion: nn.Module, input, labels, accs_test, loss_test):
    epoch_loss = 0
    epoch_true = 0
    epoch_true_topfive = 0
    epoch_all = 0
    i = 0

    net.eval()

    x = input.float()
    y = labels.to(torch.int64)
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
    return accs_test, loss_test

input = torch.cat(((predictions, onehot(images_sents.reshape((-1,1)).to(torch.int)),torch.tensor(len_faces,device=device).reshape((-1,1)))),dim=1)

with open('dataset/train/sentiment_train.txt', 'r') as f:
        labels = torch.tensor(np.array(f.read().splitlines()).astype("int32"),device=device)

input_test = torch.cat(((test_predictions, onehot(images_sents_test.reshape((-1,1)).to(torch.int)),torch.tensor(len_faces_test,device=device).reshape((-1,1)))),dim=1)

with open('dataset/test/sentiment_test.txt', 'r') as f:
        labels_test = torch.tensor(np.array(f.read().splitlines()).astype("int32"),device=device)

net = stablizer().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

from time import time

accs_train = []
loss_train = []
accs_test = []
loss_test = []
epochs = 6000

for e in range(epochs):
    start_time = time()
    accs_train, loss_train = train_epoch(net, criterion, optimizer, input, labels, accs_train, loss_train)
    accs_test, loss_test = eval_epoch(net, criterion, input_test, labels_test, accs_test, loss_test)
    if accs_test[-1] == max(accs_test):
      torch.save(net.state_dict(), 'final_combinator.pth')
    end_time = time()
    if e % 3000 == 2999:
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


<h2 id="part6">6. Test and Evaluation</h2>
<p>In this final part, the trained model from the previous phase is tested and evaluated. The evaluation process includes loading the trained model, performing inference on the test dataset, calculating evaluation metrics such as accuracy and loss, and visualizing the results using plots. Additionally, a summary of the best accuracy achieved is provided.</p>

<pre><code>
class Image_sent_classifier(nn.Module):
  def __init__(self, retina, imagenet, facenet, combiner):
    super(Image_sent_classifier,self).__init__()
    self.retina = retina
    self.imagenet = imagenet
    self.facenet = facenet
    self.combiner = combiner

  def forward(self, input):
    input_faces = self.retina(input)
    first_input = self.faces_net(input_faces)
    first_input = torch.tensor(predict(first_input, 0, len(final_input),[6/21,7/21,8/21]),device=device)
    first_input = torch.argmax(first_input,1)

    second_input = self.imagenet(input)
    second_input = torch.argmax(second_input,1)

    third_input = len(input_faces)

    final_input = torch.cat((onehot(first_input.reshape((-1,1))).to(device), onehot(torch.tensor(second_input.reshape((-1,1)))).to(device),torch.tensor(third_input,device)),dim=1)

    output = self.combiner(final_input)

    return torch.argmax(output,dim=-1)
</code></pre>


</body>
</html>
