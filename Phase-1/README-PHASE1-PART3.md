
<body>

<h1>Phase1 - Part3 : Multi-Modal Visual Sentiment Classification</h1>

<p>This repository contains code for multi-modal sentiment classification, where sentiments are predicted based on visual information. The code utilizes PyTorch for model training and evaluation.</p>

<h2>Introduction</h2>

<p>Multi-modal sentiment classification aims to predict sentiments from both text and images. This repository provides a PyTorch implementation for this task, including data preprocessing, model training, and evaluation.</p>

<h2>Dataset</h2>

<p>The dataset used in this project is the MSCTD dataset, which includes textual and visual information paired with sentiment labels. It consists of English-German pairs of sentences with corresponding sentiment labels.</p>

<h3>Data Preprocessing</h3>

<p>The dataset preprocessing involves organizing the data into appropriate directories and files. The provided code block handles data loading and preprocessing tasks, including downloading the dataset, extracting files, and organizing them into train, test, and dev sets.</p>

<pre><code class="language-python">

<span class="token comment"># Example: Loading and preprocessing the dataset</span>
<span class="token keyword">from</span> google.colab <span class="token keyword">import</span> drive
drive.mount(<span class="token string">'/content/drive'</span>)
<span class="token punctuation">%cd</span> drive/My Drive/
<span class="token comment"># ...</span>
</code></pre>

<h2>Model Architecture</h2>

<p>The model architecture consists of two main components:</p>

<ol>
    <li>Textual Sentiment Analysis</li>
    <li>Visual Sentiment Analysis</li>
</ol>

<h3>Textual Sentiment Analysis</h3>

<p>For textual sentiment analysis, we utilize a pre-trained model to extract features from text data. These features are then fed into a fully connected neural network for sentiment prediction.</p>

<pre><code class="language-python">

<span class="token comment"># Example: Defining and training the textual sentiment analysis model</span>
<span class="token keyword">class</span> TextSentimentModel(nn.Module):
    <span class="token keyword">def</span> <span class="token function">__init__</span>(<span class="token parameter">self</span>, input_dim, hidden_dim, output_dim</span>):
        <span class="token comment"># Define layers</span>
        <span class="token keyword">self</span>.fc1 <span class="token operator">=</span> nn.Linear(input_dim, hidden_dim)
        <span class="token keyword">self</span>.fc2 <span class="token operator">=</span> nn.Linear(hidden_dim, output_dim)
        <span class="token comment"># Other layers...</span>

    <span class="token keyword">def</span> forward(<span class="token parameter">self, x</span>):
        <span class="token comment"># Forward pass</span>
        x <span class="token operator">=</span> F.relu(<span class="token keyword">self</span>.fc1(x))
        x <span class="token operator">=</span> <span class="token keyword">self</span>.fc2(x)
        <span class="token keyword">return</span> x

<span class="token comment"># Instantiate the model</span>
model <span class="token operator">=</span> TextSentimentModel(input_dim, hidden_dim, output_dim)

<span class="token comment"># Define loss function and optimizer</span>
criterion <span class="token operator">=</span> nn.CrossEntropyLoss()
optimizer <span class="token operator">=</span> torch.optim.Adam(model.parameters(), lr<span class="token operator">=</span>learning_rate)

<span class="token comment"># Training loop...</span>
</code></pre>

<h3>Visual Sentiment Analysis</h3>

<p>For visual sentiment analysis, we use a pre-trained convolutional neural network (CNN) to extract features from images. These features are then combined with text features for sentiment prediction.</p>

<pre><code class="language-python">...
<span class="token comment"># Visual Sentiment Analysis Code Block</span>
<span class="token comment"># ...</span>

<span class="token comment"># Example: Defining and training the visual sentiment analysis model</span>
<span class="token keyword">class</span> VisualSentimentModel(nn.Module):
    <span class="token keyword">def</span> <span class="token function">__init__</span>(<span class="token parameter">self, input_dim, hidden_dim, output_dim</span>):
        <span class="token comment"># Define layers</span>
        <span class="token keyword">self</span>.conv1 <span class="token operator">=</span> nn.Conv2d(in_channels, out_channels, kernel_size)
        <span class="token keyword">self</span>.fc1 <span class="token operator">=</span> nn.Linear(input_dim, hidden_dim)
        <span class="token keyword">self</span>.fc2 <span class="token operator">=</span> nn.Linear(hidden_dim, output_dim)
        <span class="token comment"># Other layers...</span>

    <span class="token keyword">def</span> forward(<span class="token parameter">self, x</span>):
        <span class="token comment"># Forward pass</span>
        x <span class="token operator">=</span> F.relu(self.conv1(x))
        x <span class="token operator">=</span> F.max_pool2d(x, kernel_size)
        x <span class="token operator">=</span> x.view(-1, num_flat_features(x))
        x <span class="token operator">=</span> F.relu(self.fc1(x))
        x <span class="token operator">=</span> self.fc2(x)
        <span class="token keyword">return</span> x

<span class="token comment"># Instantiate the model</span>
model <span class="token operator">=</span> VisualSentimentModel(input_dim, hidden_dim, output_dim)

<span class="token comment"># Define loss function and optimizer</span>
criterion <span class="token operator">=</span> nn.CrossEntropyLoss()
optimizer <span class="token operator">=</span> torch.optim.Adam(model.parameters(), lr<span class="token operator">=</span>learning_rate)

<span class="token comment"># Training loop...</span>
</code></pre>

<h2>Model Training</h2>

<p>The model is trained using the combined textual and visual features. We employ a training loop that iterates over the dataset, computes predictions, calculates loss, and updates model parameters using backpropagation.</p>

<pre><code class="language-python">...
<span class="token comment"># Model Training Code Block</span>
<span class="token comment"># ...</span>

<span class="token comment"># Example: Training loop</span>
<span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token built_in">range</span>(num_epochs):
    <span class="token comment"># Forward pass</span>
    outputs <span class="token operator">=</span> model(inputs)
    loss <span class="token operator">=</span> criterion(outputs, labels)

    <span class="token comment"># Backward pass and optimization</span>
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    <span class="token comment"># Print progress</span>
    <span class="token keyword">if</span> (<span class="token variable">epoch</span><span class="token operator">+</span><span class="token number">1</span>) <span class="token operator">%</span> print_every <span class="token operator">==</span> <span class="token number">0</span>:
        <span class="token keyword">print</span>(<span class="token string">f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'</span>)
</code></pre>

<h2>Model Evaluation</h2>

<p>After training, the model is evaluated on a separate test set to assess its performance. Evaluation metrics such as accuracy are computed to measure the model's effectiveness in predicting sentiments.</p>

<pre><code class="language-python">...
<span class="token comment"># Model Evaluation Code Block</span>
<span class="token comment"># ...</span>

<span class="token comment"># Example: Evaluation loop</span>
<span class="token keyword">with</span> torch.no_grad():
    correct <span class="token operator">=</span> <span class="token number">0</span>
    total <span class="token operator">=</span> <span class="token number">0</span>
    <span class="token keyword">for</span> images, labels <span class="token keyword">in</span> test_loader:
        images <span class="token operator">=</span> images.to(device)
        labels <span class="token operator">=</span> labels.to(device)
        outputs <span class="token operator">=</span> model(images)
        _, predicted <span class="token operator">=</span> torch.max(outputs.data, <span class="token number">1</span>)
        total <span class="token operator">+=</span> labels.size(<span class="token number">0</span>)
        correct <span class="token operator">+=</span> (<span class="token variable">predicted</span> <span class="token operator">==</span> labels).sum().item()

    <span class="token keyword">print</span>(<span class="token string">f'Accuracy of the network on the test images: {100 * correct / total}%'</span>)
</code></pre>

<h2>Conclusion</h2>

<p>This README provides an overview of the multi-modal sentiment classification project using PyTorch. For further details, please refer to the code and comments provided.</p>

</body>
</html>
