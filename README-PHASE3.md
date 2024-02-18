
</head>
<body>

<h1>Phase 3: Combination two aproach of Multi-Modal sentiment</h1>
    <p>During this phase, we focus on integrating two crucial aspects of our project. The process unfolds in two main sections:</p>
    <h2>1. Preparing and Training Models for Combination:</h2>
    <ul>
        <li>In the initial section, we retrieve the pre-trained models from earlier phases.</li>
        <li>Following this, we meticulously prepare the data required for model fusion.</li>
        <li>Subsequently, we initiate the training process, where both models are combined and trained together.</li>
    </ul>
   <h2>2. Model Integration and Evaluation:</h2>
    <ul>
        <li>The second section is dedicated to merging the two models into a single, cohesive entity by defining a final model architecture.</li>
        <li>We then execute the training procedure for this unified model, aiming to enhance its performance.</li>
        <li>Finally, we conduct a comprehensive evaluation of the integrated model to assess its effectiveness.</li>
    </ul>
    <p>For a more in-depth exploration of the methodologies employed and the results obtained during this phase, please refer to the respective notebook files.</p>
<h2>Table of Contents For Section 1</h2>
<ol>
  <li><a href="#define-models">Define Models</a></li>
  <li><a href="#load-models">Load models in earlier phases</a></li>
  <li><a href="#get-embeded-vectors">Get embedded vectors from earlier phases</a></li>
  <li><a href="#define-model-architecture">Define Model architecture</a></li>
  <li><a href="#training-transformer-model">Training Transformer Model</a></li>
  <li><a href="#evaluation-transformer-model">Evaluation Transformer Model</a></li>
  <li><a href="#use-transformer-model-as-a-backbone">Use transformer model as a backbone</a></li>
  <li><a href="#preparing-dataset">Preparing dataset</a></li>
  <li><a href="#training-model">Training Model</a></li>
  <li><a href="#evaluation-model">Evaluation Model</a></li>
</ol>

<h1>Table of Contents For Section 2</h1>
<ol>
  <li><a href="#prepare-dataset">Prepare Dataset</a></li>
  <li><a href="#processing-text">Processing Text</a></li>
  <li><a href="#bert-configuration">Bert Configuration</a></li>
  <li><a href="#define-final-model">Define Final Model</a></li>
  <li><a href="#train-final-model">Train Final Model</a></li>
  <li><a href="#eval-final-model">Evaluation Final Model</a></li>
</ol>

<h2 id="define-models">1. Define Models</h2>
<p>In this section, we define models for the project.</p>
<p>The provided code defines models for BERT and EfficientNet.</p>

<h3>Bert Congfiguration</h3>
<pre><code>### Bert Congfiguration
!pip install -q transformers
from transformers import BertConfig, BertTokenizer
from transformers import BertModel, AutoModel, BertForSequenceClassification
from transformers import AdamW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# general config
MAX_LEN = 30
# Other configurations...
</code></pre>

<h3>Load models in earlier phases</h3>
<pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
# Other configurations...
</code></pre>


<h2 id="load-models">2. Load models in earlier phases</h2>
<p>This section focuses on loading models that were trained in earlier phases.</p>

<h3>Load Pretrained BERT Model</h3>
<pre><code>from transformers import BertForSequenceClassification

def load_pretrained_bert(name='models/bert_model.pt'):
    model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
        num_labels = 3,
        output_attentions = False,
        output_hidden_states = False,
    ).to(device)
    model.load_state_dict(torch.load(name))
    model.classifier = nn.Sequential()
    return model
</code></pre>

<h3>Load Pretrained Image Model (EfficientNet)</h3>
<pre><code>from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

def load_pretrained_image(name = 'models/scene_modal_en.pth'):
    # Define a class for last layer and load the pretrained model
    # Other configurations...
    return image_model
</code></pre>


<h2 id="get-embeded-vectors">3. Get Embedded Vectors from Earlier Phases</h2>
<p>In this section, we obtain embedded vectors from the models trained in earlier phases.</p>

<h3>Step 1: Load Models Trained in Earlier Phases</h3>
<pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
transform = transforms.Compose([transforms.ToTensor()
                                ,transforms.Resize((288,288),transforms.InterpolationMode("bicubic"))
                                ,transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])     
def bert_preprocess(text):
    return tokenizer.encode_plus(
        text,
        max_length= MAX_LEN,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        padding='max_length',
        return_tensors='pt',
    )
train_dataset = Final_Dataset('dataset/train', 'train_ende', 'image_index_train.txt', 'english_train.txt', 'sentiment_train.txt',preprocess_func=bert_preprocess,transform=transform)
dev_dataset = Final_Dataset('dataset/dev', 'dev', 'image_index_dev.txt', 'english_dev.txt', 'sentiment_dev.txt',preprocess_func=bert_preprocess,transform=transform)
test_dataset = Final_Dataset('dataset/test', 'test', 'image_index_test.txt', 'english_test.txt', 'sentiment_test.txt',preprocess_func=bert_preprocess,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
</code></pre>

<h3>Step 2: Get Embedded Vectors of Our Data Using Previous Phases' Models</h3>
<pre><code>import tqdm
import pickle

vectors = []
labels = []
with tqdm.tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
      with torch.no_grad():  
        for i, m in pbar:
            data_i,image_i, y = m
            (input_ids, attention_mask, token_type_ids) = data_i.values()
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            y = y.to(device)
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            token_type_ids = token_type_ids.squeeze(1)
            output = text_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            rep_text = output.logits
            y = y.to(device)
            image_i = image_i.to(device)
            rep_image = image_model(image_i)
            rep = torch.cat((rep_text,rep_image),dim=1)
            print(rep.size())
            print(y.size())
            torch.save(y, "dataset/train/cated_data/labels_"+str(i)+".pt") 
            torch.save(rep, "dataset/train/cated_data/vectors_"+str(i)+".pt") 
</code></pre>


<h2 id="define-model-architecture">4. Define Model Architecture</h2>
<p>In this section, we define the architecture of our multi-modal model.</p>

<h3>MultiModalModel Definition</h3>
<pre><code>import torch.nn.functional as F

class MultiModalModel(nn.Module):
    def __init__(self, num_input, num_classes=3):
        super(MultiModalModel, self).__init__()
        hidden_1 = num_input // 2
        self.fc1 = nn.Linear(num_input, hidden_1)
        self.fc2 = nn.Linear(hidden_1, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
</code></pre>


<h2 id="training-transformer-model">5. Training Transformer Model</h2>
<p>In this section, we perform the training of the transformer model.</p>

<h3>Training Configuration</h3>
<pre><code>LEARNING_RATE = 1e-5
EPOCH = 30
BATCH_SIZE = 64
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, modalmodel.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)
criterion = nn.CrossEntropyLoss()
</code></pre>

<h3>Training Model</h3>
<pre><code>modalmodel, min_val_loss = train_model(modalmodel, (data_train, labels_train), (data_test, labels_test), EPOCH, criterion, optimizer, model_name='multi_modal', scheduler=scheduler, batch_size=BATCH_SIZE)
</code></pre>


<h2 id="evaluation-transformer-model">6. Evaluation Transformer Model</h2>
<p>This section focuses on evaluating the performance of the transformer model.</p>

<h3>Evaluation</h3>
<pre><code>def eval_model(model, loader, metrics=metrics, set_name='Test', plot_confusion_matrix=True):
    eval_result = one_epoch(model, loader, criterion, train=False, set_name=set_name, metrics=metrics)
    disp = ConfusionMatrixDisplay(eval_result.pop('confusion_matrix'))
    if plot_confusion_matrix:
        disp.plot()
    return eval_result

model.load_state_dict(torch.load('models/multi_modal.pt'))
eval_model(model, test_loader)
</code></pre>


<h2 id="use-transformer-model-as-a-backbone">7. Use Transformer Model as a Backbone</h2>
<p>In this section, we utilize a transformer model as a backbone for our task.</p>

<h3>Essential Imports</h3>
<pre><code>!pip install -q transformers
from transformers import AutoTokenizer, VisualBertForVisualReasoning
from transformers import BertConfig, BertTokenizer
from transformers import BertModel, AutoModel, BertForSequenceClassification
from transformers import AdamW
import torch
from torch.nn import BCEWithLogitsLoss as logit_bce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
</code></pre>

<h3>Model Configuration</h3>
<pre><code>device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VIS_EMB_DIM = 1408
TXT_EMB_DIM = 768

TRAIN_DATA_SIZE = 40
TEST_DATA_SIZE = 10

TRAIN_BATCH_SIZE = 240
TEST_BATCH_SIZE = 240

model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2",
                                                    num_labels=3,
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    visual_embedding_dim=VIS_EMB_DIM,
                                                    ignore_mismatched_sizes=True)

model = model.to(device)
</code></pre>


<h2 id="preparing-dataset">8. Preparing Dataset</h2>
<p>In this section, we prepare the dataset suitable for the transformer model.</p>

<h3>Dataset Preparation</h3>
<pre><code>class Final_Emb_Dataset(Dataset):
    def __init__(self, dataset_dir, visual_embedded_dir, visual_file_len, transform=None):
        # Dataset initialization code
        # Other configurations...
        pass

    def __len__(self):
        # Define length of the dataset
        pass

    def __getitem__(self, idx):
        # Define how to get an item from the dataset
        pass

transform = T.Compose([T.ToTensor()])

trainset = Final_Emb_Dataset('dataset/train', 'cated_data', TRAIN_DATA_SIZE, transform=transform)
testset = Final_Emb_Dataset('dataset/test', 'cated_data', TEST_DATA_SIZE, transform=transform)
train_loader = DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False)
</code></pre>


<h2 id="training-model">9. Training Model</h2>
<p>In this section, we train the transformer model using the prepared dataset.</p>

<h3>Training Configuration</h3>
<pre><code>LEARNING_RATE = 1e-4
EPSILON = 5e-8
EPOCHS = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)
criterion = logit_bce()
</code></pre>

<h3>Training Model</h3>
<pre><code>model, min_val_loss = train_model(model, (train_loader, test_loader), EPOCHS, criterion, optimizer, model_name='visualbert-model', scheduler=None)
</code></pre>


<h2 id="evaluation-model">10. Evaluation Model</h2>
<p>This section focuses on evaluating the performance of the trained transformer model.</p>

<h3>Evaluation</h3>
<pre><code>def eval_model(model, loader, metrics=metrics, set_name='Test', plot_confusion_matrix=True):
    results = one_epoch(model, loader, criterion, train=False, set_name=set_name, metrics=metrics)
    disp = ConfusionMatrixDisplay(results.pop('confusion_matrix'))
    if plot_confusion_matrix:
        disp.plot()
    return results

model.load_state_dict(torch.load('models/visualbert-model.pt'))
eval_model(model, test_loader)
</code></pre>

<h2 id="prepare-dataset">1. Prepare Dataset</h2>
<p>In this part, we start by loading the dataset from a CSV file using pandas. We then define the device to be used (either CUDA GPU if available or CPU). Constants such as batch sizes, maximum length, epochs, learning rate, and model name are also defined. Additionally, we define a custom dataset class <code>MSCTD_Dataset</code> for handling the dataset, and transform functions for image data.</p>
<pre><code># Bert configuration
train_size = int(0.3 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = int(0.1 * len(dataset))
train_size, val_size, test_size
train_set, val_set, test_set, _ = torch.utils.data.random_split(dataset, [train_size, val_size, test_size, len(dataset) - train_size - val_size - test_size])
train_loader = DataLoader(train_set, batch_size=60, shuffle=True)
val_loader = DataLoader(val_set, batch_size=60, shuffle=False)
test_loader = DataLoader(test_set, batch_size=60, shuffle=False)
</code></pre>

<h2 id="processing-text">2. Processing Text</h2>
<p>Here, we perform preprocessing steps for the text data. This includes tokenization, removing punctuation, removing emojis, removing stopwords, lemmatization, handling numbers, and handling unknown words using the <code>pyenchant</code> library.</p>
<pre><code>from transformers import BertForSequenceClassification

#Load pretrained BERT model
def load_pretrained_bert(name='bert_model.pt'):
    # Load pretrained BERT model
    # ...
    return model

#Define last layer for multimodal fusion
class lastLayer(nn.Module):
    # Definition of last layer for multimodal fusion
    # ...

#Load pretrained image model
def load_pretrained_image(name ='scene_modal_en.pth'):
    # Load pretrained image model
    # ...
    return image_model
</code></pre>


<h2 id="bert-configuration">3. Bert Configuration</h2>
<p>This part involves setting up the configuration for BERT. We split the dataset into train, validation, and test sets, define data loaders for each set, and perform BERT-specific preprocessing using the <code>BertTokenizer</code>.</p>
<pre><code>from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

#Training configuration
#...

#Define evaluation function
def eval_model(models, loader, metrics=metrics, set_name='Test', plot_confusion_matrix=True):
    # Evaluation function definition
    # ...

# Before training evaluation
eval_model((text_model, image_model), test_loader)

#Training
EPOCH = 1
image_model, min_val_loss = train_model((text_model, image_model), (train_loader, val_loader), EPOCH, criterion, optimizer, model_name='drive/MyDrive/Deep Project/weakly_sup', scheduler=scheduler)

#Save trained models
torch.save(image_model.state_dict(), 'drive/MyDrive/Deep Project/weak_sup.pt')
torch.save(text_model.state_dict(), 'drive/MyDrive/Deep Project/bert_model_weak_sup.pt')

#Evaluation after training
eval_model((text_model, image_model), val_loader)
eval_model((text_model, image_model), test_loader)
</code></pre>


<h2 id="define-final-model">4. Define Final Model</h2>
<p>We define the final model architecture, consisting of a BERT model for text processing and an image model (EfficientNet-B2) for image processing. The last layer of the image model is replaced with a custom fully connected layer for multimodal fusion.</p>
<code><pre>
# Training
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tqdm

def one_epoch(models, loader, criterion, optimizer=None, epoch='', train=True, set_name='Train', metrics=None):
    # Function to perform one epoch of training or evaluation

def train_model(models, dataloaders, num_epochs, criterion, optimizer, model_name='pytroch-model', scheduler=None):
    # Function to train the model over multiple epochs

# Define hyperparameters and initialize the models
LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, image_model.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=0.5)
criterion = DistillationLoss(alpha=0.5)
EPOCH = 1  # Number of epochs for training

# Train the model
image_model, min_val_loss = train_model((text_model, image_model), (train_loader, val_loader), EPOCH, criterion, optimizer, model_name='drive/MyDrive/Deep Project/weakly_sup', scheduler=scheduler)

# Save the trained models
torch.save(image_model.state_dict(), 'drive/MyDrive/Deep Project/weak_sup.pt')
torch.save(text_model.state_dict(), 'drive/MyDrive/Deep Project/bert_model_weak_sup.pt')

# Evaluate the trained model
eval_model((text_model, image_model), val_loader)
eval_model((text_model, image_model), test_loader)
</code></pre>

<h2 id="train-final-model">5. Train Final Model</h2>
<p>Next, we train the combined model on the prepared dataset. We define training and evaluation functions, including metrics like accuracy, precision, recall, F1-score, and confusion matrix. The model is trained using an Adam optimizer with a learning rate scheduler, and the trained models are saved for later use. Evaluation is performed before and after training to assess the model's performance.</p>
<code><pre>
# Before training evaluation
eval_model((text_model, image_model), test_loader)

# Define hyperparameters and initialize the models for training
LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, image_model.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=0.5)
criterion = DistillationLoss(alpha=0.5)
EPOCH = 1  # Number of epochs for training

# Train the model
image_model, min_val_loss = train_model((text_model, image_model), (train_loader, val_loader), EPOCH, criterion, optimizer, model_name='drive/MyDrive/Deep Project/weakly_sup', scheduler=scheduler)

# Save the trained models
torch.save(image_model.state_dict(), 'drive/MyDrive/Deep Project/weak_sup.pt')
torch.save(text_model.state_dict(), 'drive/MyDrive/Deep Project/bert_model_weak_sup.pt')

# After training evaluation
eval_model((text_model, image_model), val_loader)
eval_model((text_model, image_model), test_loader)
</code></pre>

<h2 id="eval-final-model">6. Evaluation Final Model</h2>
<p>
  Finally, This part provides a comprehensive assessment of the final integrated model's performance on the test dataset, allowing you to gauge its effectiveness in making predictions based on both text and image inputs.
</p>
  <code><pre>
#Evaluation of the final model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_final_model(models, test_loader):
    text_model, final_model = models
    final_model.eval()
    text_model.eval()

    Y_true = []
    Y_pred = []

    with torch.no_grad():
        for images, texts in test_loader:
            images = images.to(device)
            texts = texts.to(device)

            # Pass the images through the final model
            predictions = final_model(images, texts)

            # Get the predicted labels
            _, predicted = torch.max(predictions, 1)

            # Append true and predicted labels for evaluation
            Y_true.extend(texts.cpu().numpy())
            Y_pred.extend(predicted.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, average='weighted')
    recall = recall_score(Y_true, Y_pred, average='weighted')
    f1 = f1_score(Y_true, Y_pred, average='weighted')
    confusion_mat = confusion_matrix(Y_true, Y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_mat)
    disp.plot()

    # Return evaluation metrics
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'confusion_matrix': confusion_mat
    }

    return evaluation_results

#Evaluate the final model using the test loader
final_model_evaluation = evaluate_final_model((text_model, final_model), test_loader)
print("Final Model Evaluation Results:")
print(final_model_evaluation)

</code></pre>


</body>
</html>
