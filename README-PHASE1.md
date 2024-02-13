<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Multi-Modal Sentiment Classification with PyTorch</h1>

<p>This repository contains code for multi-modal sentiment classification using PyTorch. The project involves predicting sentiments from both textual and visual information. It includes data preprocessing, model architecture, training, and evaluation.</p>

<h2>Dataset</h2>

<p>The dataset used in this project is the MSCTD dataset, which contains English-German pairs of sentences paired with sentiment labels. The data is preprocessed and organized into train, test, and dev sets.</p>

<h2>Model Architecture</h2>

<p>The model architecture consists of two main components:</p>

<ol>
    <li>Textual Sentiment Analysis</li>
    <li>Visual Sentiment Analysis</li>
</ol>

<h3>Textual Sentiment Analysis</h3>

<p>For textual sentiment analysis, a fully connected neural network is used. Textual features are extracted using a pre-trained model and fed into the neural network for sentiment prediction.</p>

<h3>Visual Sentiment Analysis</h3>

<p>Visual sentiment analysis involves utilizing a pre-trained convolutional neural network (CNN) to extract features from images. These features are combined with textual features for sentiment prediction.</p>

<h2>Model Training</h2>

<p>The model is trained using the combined textual and visual features. A training loop iterates over the dataset, computing predictions, calculating loss, and updating model parameters using backpropagation.</p>

<h2>Model Evaluation</h2>

<p>After training, the model is evaluated on a separate test set to assess its performance. Evaluation metrics such as accuracy are computed to measure the effectiveness of the model in predicting sentiments.</p>

<h2>Usage</h2>

<p>To train the model, follow these steps:</p>

<ol>
    <li>Preprocess the dataset using the provided code.</li>
    <li>Define and train the textual sentiment analysis model.</li>
    <li>Define and train the visual sentiment analysis model.</li>
    <li>Combine both models for multi-modal sentiment classification.</li>
    <li>Evaluate the trained model on the test set.</li>
</ol>

<h2>Contributing</h2>

<p>Contributions to this project are welcome. Feel free to open an issue or submit a pull request.</p>

<h2>License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
