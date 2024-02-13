<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Multilingual Sentiment Analysis and Face Recognition</h1>

    <p>This repository contains code for a project involving sentiment analysis on multilingual text data and face recognition using CNN models.</p>

    <h2>Installation</h2>
    <p>To run the code, follow these steps:</p>
    <ol>
        <li>Mount Google Drive:</li>
        <pre><code>from google.colab import drive
drive.mount('/content/drive')
%cd drive/My Drive/</code></pre>
        <li>Clone the repository:</li>
        <pre><code>! git clone https://github.com/XL2248/</code></pre>
        <li>Install required dependencies:</li>
        <pre><code>!pip install --upgrade --no-cache-dir gdown
!pip install mtcnn
!pip install git+https://github.com/elliottzheng/face-detection.git@master
!pip install einops --upgrade</code></pre>
    </ol>

    <h2>Data Preparation</h2>
    <p>The data is organized into train, test, and dev sets in the 'dataset' directory.</p>

    <h3>MSCTD Dataset</h3>
    <p>The MSCTD dataset is used for sentiment analysis, consisting of English-German parallel text data along with image indices and sentiment labels.</p>

    <h3>Face Dataset</h3>
    <p>A custom dataset is created for face recognition, with images stored in the 'train/faces' and 'test/faces' directories.</p>

    <h2>Usage</h2>
    <p>After preparing the data, the code can be executed to train and evaluate the models.</p>

    <h3>Phase 1: Localization and CNN Model</h3>
    <p>The code comprises the following phases:</p>
    <ol>
        <li>Face Extraction: Utilizes RetinaFace for face detection and extracts faces from images.</li>
        <li>CNN Model Training: Trains a CNN model using EfficientNet-B2 architecture for face recognition.</li>
    </ol>

    <h2>Results</h2>
    <p>The model's performance is evaluated based on loss and accuracy metrics.</p>

    <h3>Face Model Loss</h3>
    <p><img src="path/to/face_model_loss_plot" alt="Face Model Loss Plot"></p>

    <h3>Face Model Accuracy</h3>
    <p><img src="path/to/face_model_accuracy_plot" alt="Face Model Accuracy Plot"></p>

    <h2>Contributing</h2>
    <p>Contributions are welcome. For major changes, please open an issue first to discuss potential improvements.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

    <h2>Python Code</h2>
    <pre><code>main_detector = RetinaFace()
def face_detector(image):
    faces_boundaries = main_detector(image)
    faces = []
    for i in range(len(faces_boundaries)):
      stats, _, score = faces_boundaries[i]
      stats = stats.astype(int)
      if score>0.95:
        faces.append(Image.fromarray(image[max(0,stats[1]):min(image.shape[0],stats[3]),
                                           max(0,stats[0]):min(image.shape[1],stats[2])]))
    return faces

class Face_Dataset_eval(Dataset):
  def __init__(self, images,trasnform):
    self.images = images
    self.transform = transform
    
  def __len__(self):
        return len(self.images)


  def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
def statical_random_generator(counter):
  r = np.random.random(1)
  if r<counter[0]:
    return 0
  elif r<(counter[0]+counter[1]):
    return 1
  else:
    return 2
def predict_mode(image,transform,counter):
  predictions = []
  faces = face_detector(np.array(image.convert('RGB')))
  dataset_face_eval = Face_Dataset_eval(faces,transform)
  if (len(faces)==0):
    return 1
  testface_loader = torch.utils.data.DataLoader(dataset_face_eval, batch_size=len(faces), shuffle=False, num_workers=2)
  for x in testface_loader:
      predictions = net(x.to(device)).float().argmax(-1)
  if 0 in predictions and not  2 in predictions:
    return 0
  elif 2 in predictions and not  0 in predictions:
    return 2
  elif 2 in predictions or 0 in predictions:
    return statical_random_generator(counter)
  else:
    return 1
labels = np.vectorize(lambda t: ['Neutral' , 'Negative' , 'Positive'][t])(testset.sentiments)
m = {'Neutral':0,'Negative':1,'Positive':2}
lab_set = [m[k] for k in labels]
counter=[0,0,0]
for lab in tqdm(lab_set):
  counter[lab]+=1
from collections import Counter
from tqdm import tqdm
from torchvision import transforms
truth = []
predict = []
net.load_state_dict(torch.load("models/face_modal.pth",map_location=torch.device('cpu')))
transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Resize((288,288),transforms.InterpolationMode("bicubic"))
                                ,transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                                
counter = np.array(counter)
counter = np.divide(counter,np.sum(counter))
for i in tqdm(range(len(testset)),total = len(testset)-4501):
  _,image,_ = testset[i]
  p = predict_mode(image,transform,counter)
  truth.append(sentiment)
  predict.append(p)
# pbar.set_description(f"Accuracy: {np.sum(np.array(truth) == np.array(predict))/len(truth)*100:.2f}%")
  if i%500== 0 and i != 0:
    with open("models/Truth_results_face_without_augment"+str(int(i/500))+".txt", 'wb') as f:
          pickle.dump(truth, f)
    with open("models/Pred_results_face_without_augment"+str(int(i/500))+".txt", 'wb') as f:
          pickle.dump(predict, f)
    truth = []
    predict = []
with open("models/Pred_results_face_without_augment11.txt", 'wb') as f:
      pickle.dump(predict, f)

with open("models/Truth_results_face_without_augment11.txt", 'wb') as f:
      pickle.dump(truth, f)
# test_data = pickle.load(open('test_data.pkl', 'rb'))
# # Evaluation
# model.eval()
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print('Accuracy of the network on the test images: %d %%' % (
#     100 * correct / total))
</code></pre>

</body>
</html>
