<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Multi-Modal Visual Sentiment Classification</h1>

<p>This phase contains code for multi-modal visual sentiment classification using PyTorch. The project involves predicting sentiments from visual information. It includes data preprocessing, model architecture, training, and evaluation.</p>

<h2>Dataset</h2>

<p>The dataset used in this phase is the MSCTD dataset, which contains English-German pairs of sentences paired with sentiment labels. The data is preprocessed and organized into train, test, and dev sets. Atteniton that we only use visual information of this dataset in this phase.</p>

<h2>Model Architecture</h2>

<p>The model architecture consists of four main components:</p>

<ol>
    <li>Face Extraction</li>
    <li>Face Analysis</li>
    <li>Overal Image Analysis</li>
    <li>Combine Results</li>
</ol>

<h3>Face Extraction</h3>

<p>The procedure for extracting faces from images involves several steps. First, the input image is passed through a face detection model, such as RetinaFace, which identifies potential faces in the image along with their bounding boxes. Then, each bounding box is used to crop the corresponding region from the original image, resulting in individual face images. These cropped face images are then typically resized to a standard size and format for further processing or analysis. Finally, the extracted face images can be used for various tasks such as facial recognition, sentiment analysis, or image classification.</p>

<h3>Face Analysis</h3>

<p>After extracting the faces from images, the face analysis method typically involves several stages of processing. First, the extracted face images may undergo preprocessing steps such as resizing, normalization, and transformation to ensure uniformity and enhance the quality of the input data. Following preprocessing, the face images are often fed into a deep learning model or a pre-trained neural network designed for facial analysis tasks. This model can perform various analyses on the face images, including facial expression recognition, age and gender estimation, facial landmark detection, or even identity verification. The output of the model provides valuable insights into the characteristics and attributes of the detected faces, enabling applications such as emotion detection, audience sentiment analysis, or personalized user experiences. Overall, face analysis methods leverage advanced machine learning techniques to extract meaningful information from facial images, contributing to a wide range of applications in computer vision and artificial intelligence.</p>

<h3>Overal Image Analysis</h3>

<p>After the face extraction process, the entire image analysis involves several key steps aimed at understanding the context and sentiment conveyed by the image as a whole. Firstly, the preprocessed images are passed through a convolutional neural network (CNN) or another deep learning model trained on large-scale image datasets such as ImageNet. This model extracts high-level features from the images, capturing patterns, objects, and scene characteristics. Subsequently, the extracted features are utilized as input to another neural network, typically a multi-layer perceptron (MLP) or a fully connected network, which predicts the sentiment or emotion associated with the image. The sentiment prediction encompasses various aspects, such as positive, negative, or neutral emotions or sentiments, which are typically represented as probability distributions over different sentiment classes. Finally, the predicted sentiment provides insights into the overall mood, tone, or message conveyed by the image, enabling applications such as sentiment analysis in social media, image classification, or content moderation. Overall, the whole image analysis pipeline combines feature extraction, deep learning, and sentiment prediction techniques to gain a comprehensive understanding of the sentiment and context encapsulated within the images.</p>

<h3>Combine Result</h3>

<p>Combining face analysis with whole image analysis involves synthesizing sentiments predicted for individual faces with the broader sentiment gleaned from the entire image. Initially, sentiments assigned to each face are aggregated to form an overall representation, potentially weighted by factors like face prominence or size. Concurrently, sentiments derived from the image as a whole are considered, capturing the broader emotional context. Finally, these two sets of sentiments are fused using methods like weighted averaging or neural network ensembling, resulting in a comprehensive sentiment representation that accounts for both individual facial expressions and the overall image sentiment. This integration enhances understanding of the image's emotional content, enabling applications like sentiment-aware image retrieval and content recommendation.</p>

<h2>Model Training</h2>

<p>The model is trained using the combined visual features. A training loop iterates over the dataset, computing predictions, calculating loss, and updating model parameters using backpropagation.</p>

<h2>Model Evaluation</h2>

<p>After training, the model is evaluated on a separate test set to assess its performance. Evaluation metrics such as accuracy are computed to measure the effectiveness of the model in predicting sentiments.</p>

<h2>Usage</h2>

<p>To train the model, follow these steps:</p>

<ol>
    <li>Preprocess the dataset using the provided code. (For example Augmentation)</li>
    <li>Extracting Faces in images for train the model based on faces.</li>
    <li>Preprocess the dataset which maded by extracting the faces.</li>
    <li>Define and train the visual sentiment analysis model based on faces.</li>
    <li>Define and train the visual sentiment analysis model bease on overal features of image.</li>
    <li>Combine both models for multi-modal visual sentiment classification.</li>
    <li>Evaluate the trained model on the test set.</li>
</ol>

<h2>Contributing</h2>

<p>Contributions to this project are welcome. Feel free to open an issue or submit a pull request.</p>

</body>
</html>
