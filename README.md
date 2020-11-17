# BBC NLP Problem

### Author: John Case

This was a fun project.  It came at a time when I wanted to work on a few other things that I hadn't had time for previously.  We have a few T7910 towers in the department that are underutilized, but could be a great asset for data science work if configured properly.  I also found a GeForce GTX 1080 GPU laying around in our IT shop, which would be great to use for local DL network training.  I had previously done a lot of deep learning work in Keras/tensorflow but was unhappy with runtime durations when testing different models and hyperparameter searches.  PyTorch has been gaining momemtum due to computation efficiencies and I wanted to learn how to use it.  This project came at a time when I was ready to try to improve on these things, so I included these efforts as part of this project:

1. Install linux on one of the underutilized Windows towers and use it as a remote data science resource.
2. Install a GPU and configure for deep learning.
3. Learn to use PyTorch.

All in all, this was a good experience.  I have of course previously used linux and GPUs to train neural networks, but I had not done all the installs and configurations myself.   

<hr />

### Setting up the remote data science machine

The T7910 tower I grabbed is a very capable machine.  It has 386Gb RAM and dual CPUs with 16 cores each.  This was a great option for dedicated data science work, but the machine had been sitting idle for some time.  My plan was to install Linux, RStudio Server and Anaconda for data science work and then provide logins for anyone in the department who wanted access.  Jupyter notebooks in Anaconda or RStudio Server could just be port forwarded to their working machine when they wanted to do any dedicated data science. 

To tackle the hardware configurations, I installed Ubuntu 20.04 from a boot stick and then completed the system installs to make this a useful data science machine for others in the department.  To do this, I just configured the machine for SSH on our internal WREN network.  This way anyone in the department can VPN to WREN and SSH into the machine to complete data science tasks. I had been using port forwarding to work on a remote machine and found it very easy and useful. 

<hr />

### Installing/configuring the GPU

This consisted of making physical space in the tower for the GPU, which is a 10x5 inch 2.5 slotter and fills the available space in the tower frame.  Installing the proper NVIDIA drivers, configuring the GPU as the primary device and using the existing smaller GPU purely for screen display was fairly straightforward.  Installing the CUDA developer tool kit and the linux distribution-specific CUDA libraries was the most complicated part.  In the end, I pieced together multiple blogs with the CUDA documentation to get the device operational.  

<hr />

### Learning Pytorch

I am very comfortable in Keras/Tensorflow, but was ready to improve on the dev iteration timeline that is so limiting in Keras.  The main effort in learning PyTorch was the learning the different data construct and the pure-Python development of the neural network classes and training/testing loops.  There is clearly a larger audience using Keras, because the amount of PyTorch projects available to implement and learn from are more limited.  Luckily, the PyTorch documentation is very good and I was able to do a lot straight from the docs. In the end, I am likely going to leave Keras/Tensorflow behind and shift all future projects into PyTorch. 

<hr />

### BBC NLP Project 

#### Modeling Approach

My approach to this project was to follow a path I had used in previous NLP research as part of an ORCEN project:

1. Preprocess the data.  The intent here was to reduce varaibility in the text data as much as possible without losing information used to accurately map the text to the document class. 
	a. Make the text uniformly lowercase.
	b. Remove special characters.
	c. Replace contractions with expanded words.
	d. Remove large numbers.
	e. Remove stop words.
	f. Cut the documents down to even lengths.

2. Split the data into train, validation, and test sets.  Ensure to stratify random splits by class so that the model would evenly balance improving error across the classes.

3. Build the tensor-optimized datasets.  Use the PyTorch library to setup batched tensor data for training on the newly configured GPU.  

4. Define the neural network model to be trained.  Select the appropriate loss function and optomizer.

5. Define the training and testing loops.

6. Train the model on GPU.  This should result in a significant speedup on training run iteration and allow for faster development on the model architecture and hyperparameters. 

7. Evaluate the model's performance.  It would be important to evaluate model accuracy across the different classes to understand where the model is strong or weak.  

The implementation and discussion of these steps is shown in the jupyter notebook file [AFC_NLP_v1]: (https://github.com/jjc44/BBC_NLP_project/blob/main/AFC_NLP_v1.ipynb "AFC_NLP_v1").

#### Modeling Results

##### General 

The model trained to an 89% accuracy on the test set, which is high given the small training data set.  This model does not leverage pretrained embeddings, which would lead me to believe it would need a lot of training examples to generalize well to new samples in the test set.  However, this model performed very well on the test set.  The training time of the model was essentially less than a second per epoch, which is extremely fast. 

##### Performance Metrics 

The confusion matrix is instructive into the model's performance across the classes.  The normalized confusion matrix shows the recall metric as the diagonal value (true positives / true positives + false negatives).  Precision is not directly shown, but is found by the diagonal element divided by the sum of the column values (true positives / true positives + false positives) and is calculated above.  

The confusion matrix is shown below:

![Confusion Matrix](/norm_conf_mx.png)


Precision and recall give different information about the model performance. For this use case it seems that f1 score, the harmonic mean of precision and recall, is the most important metric for the model.  

Because precision is high and recall is lower for the entertainment class, we can see that the model is under-tagging documents with this label. Therefore, errors for the true-labeled entertainment documents should be examined in more detail. If a class had high recall but suffered from lower precision, the indicates that it was over-tagging articles as being business related. 

The calculated precision, recall, and F1 metrics are shown below:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business</td>
      <td>0.891068</td>
      <td>0.904867</td>
      <td>0.897914</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Entertainment</td>
      <td>0.804598</td>
      <td>0.942761</td>
      <td>0.868217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Politics</td>
      <td>0.875000</td>
      <td>0.891599</td>
      <td>0.883221</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sports</td>
      <td>0.976087</td>
      <td>0.921971</td>
      <td>0.948258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tech</td>
      <td>0.897507</td>
      <td>0.812030</td>
      <td>0.852632</td>
    </tr>
  </tbody>
</table>

Some classes may be more likely to be confused by the model, in this case business.  This is expected because business inherently requires a business sector, which could include tech or sports or entertainment.  Additionally, politics can legislate to regulate business, providing many sources of confusion for the model. 

Intuitively, many news articles will touch on more than one topic and there may be articles that are focused evenly on two or more topics.  I can think of many instances in which a business article would contain many references to entertainment or tech.  It is reasonable that the model would confuse these related classes.  We would not expect there to be much confusion between sports, politics, and tech because there is likely little crossover among these topics in the news.  

##### Error Analysis

Generally, the entertainment class was mislabeled the most.  The highest confused class categories should be explored.  The model made ~10% of its sub-class errors when the model predicted entertainment but the article's true label was tech.


