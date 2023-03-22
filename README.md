# Multimodal Emotion Recognition
Final project for the 2022-2023 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Álvaro García Hernández**, **Carla Campàs Gené**, **Hector Colado** and **Jorge Garcia**. 

Advised by **Carlos Escolano**.


Table of Contents
=================

  * [INTRODUCTION AND MOTIVATION](#introduction-and-motivation)
  * [GOALS](#goals)
  * [BACKGROUND](#background)
  * [DATASET](#dataset)
  * [DATASET PREPROCESSING AND PREPARATION](#dataset-preprocessing-and-preparation)
    * [Text](#text)
    * [Speech](#speech)
  * [TEXT ARCHITECTURE](#text-architecture)
	  * [Simple Multilayer Perceptron (MLP) Classifier](#simple-multilayer-perceptron-mlp-classifier)
	  * [Long Short Term Memory](#long-short-term-memory)
    * [Context Model](#context-model)
    * [Transformer](#transformer)
  * [SPEECH ARCHITECTURE](#speech-architecture)
    * [2D CNN Classifier](#2d-cnn-classifier)
    * [1D CNN Classifier](#1d-cnn-classifier)
    * [LSTM Classifier](#lstm-classifier)
  * [MULTIMODAL ARCHITECTURE](#multimodal-architecture)
  * [TEXT EXPRIMENTATION](#text-experimentation)
    * [Utterance-level Classification](#utterance-level-classification)
      * [MLP](#mlp)
      * [LSTM](#lstm)
      * [TRANSFORMERS](#transformers)
    * [Dialogue-level Classification](#dialogue-level-classification)
  * [SPEECH EXPERIMENTATION](#speech-experimentation)
    * [Utterance-level Speech Classification](#utterance-level-speech-classification)
    * [Dialogue-level Speech Classification](#dialogue-level-speech-classification)
  * [MULTIMODAL EXPERIMENTATION](#mutlimodal-experimentation)
  * [CONCLUSIONS](#conclusions)
  * [FUTURE WORK](#future-work)

## INTRODUCTION AND MOTIVATION
Multimodal speech recognition is a specific application of multimodal data analysis that combines different channels of information to accurately recognize emotions from speech. This involves analyzing not only the acoustic features of speech but also the visual and textual cues, such as facial expressions and spoken words, to provide a more comprehensive understanding of the emotional state of the speaker.

One of the key motivations for multimodal speech recognition is its potential to improve human-computer interaction. By accurately recognizing a user's emotional state, a computer or virtual assistant can adapt its responses and interactions to better suit the user's needs, leading to a more natural and intuitive user experience.

Furthermore, multimodal speech recognition has the potential to enhance mental health diagnosis by providing a more objective and accurate assessment of a patient's emotional state. This can help clinicians make more informed decisions about treatment and therapy.

Finally, multimodal speech recognition can also be used to address societal issues, such as social inequality and communication barriers. By accurately recognizing emotions from speech, it can help improve communication between individuals from different cultures or with different communication styles, ultimately leading to more inclusive and diverse societies.

Overall, multimodal speech recognition is a rapidly growing field with tremendous potential for impact in various domains. By combining multiple channels of information, it offers a more accurate and nuanced understanding of human emotions, which can lead to improvements in a wide range of applications.

## GOALS
For this project, we wanted to explore multimodal data analysis from a scientific perspective. Our main goals are descibed below:
  * Understanding State-of-the-Art Multimodal Research
    * Extensively research state-of-the-art applications of multimodal architectures
    * Find accessible data pertaining to emotion detection in a multimodal scope

  * Implementing and Evaluating Fully-Functional Deep Learning Architecture able to Classify Emotions
    * Experiment with different Deep Learning Architectures for Emotion Recognition
    * Learn how to integrate multimodal data
    * Learn and experiment with augmented data (stretch goal)

## BACKGROUND

"Context-Dependent Domain Adversarial Neural Network for Multimodal Emotion Recognition" is a research paper that proposes a novel deep learning approach for recognizing emotions from multimodal data such as facial expressions, speech, and text. The proposed method is based on a domain adversarial neural network (DANN) that is capable of learning to extract features from different modalities and fuse them in a context-dependent manner to improve the overall performance of the emotion recognition system.

The DANN model is trained using a combination of labeled and unlabeled data from multiple domains, which helps the model to learn robust representations that are invariant to domain shifts. The proposed method is evaluated on several publicly available datasets, and the results show that it outperforms existing state-of-the-art methods for multimodal emotion recognition.

Similar research papers that address the topic of multimodal emotion recognition include "Multimodal Emotion Recognition Using Deep Learning: A Review" by Li et al. (2020), which provides a comprehensive review of the latest deep learning approaches for multimodal emotion recognition. "Deep Multimodal Fusion by Channel Exemplar Mining for Emotion Recognition" by Zhang et al. (2018) proposes a deep fusion approach that combines information from multiple modalities using a channel exemplar mining method. "Multi-modal Emotion Recognition using Audio and Video Modality with Deep Learning Techniques" by Sharma et al. (2020) presents a deep learning-based approach for recognizing emotions from both audio and video modalities.

Multimodal Emotion Recogntion:
  * Deep Learning Models: Most of the state-of-the-art systems use deep learning models such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and their variants such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. These models are capable of learning complex features from multimodal data such as speech, facial expressions, and text.

  * Fusion Methods: There are several methods for fusing information from different modalities such as early fusion, late fusion, and hybrid fusion. Early fusion involves concatenating the features from different modalities and feeding them to a single deep learning model, whereas late fusion involves training separate deep learning models for each modality and combining their outputs. Hybrid fusion is a combination of both early and late fusion methods.
  
  * Preprocessing Techniques: Before feeding the multimodal data to deep learning models, it is often preprocessed to extract useful features. For example, in speech-based emotion recognition, Mel-Frequency Cepstral Coefficients (MFCCs) are commonly used to represent the spectral characteristics of speech signals. In image-based emotion recognition, facial landmarks and Action Units (AUs) are extracted to capture facial expressions.

  * Datasets: There are several publicly available datasets for multimodal emotion recognition, such as the AffectNet, EmoReact, and IEMOCAP datasets. These datasets contain multimodal data such as speech, facial expressions, and text, along with annotations of different emotional states.

  * Evaluation Metrics: The performance of multimodal emotion recognition systems is often evaluated using metrics such as accuracy, F1-score, and confusion matrix. In some cases, additional metrics such as Arousal-Valence correlation and emotion recognition rate are also used to evaluate the system's performance.

Overall, multimodal emotion recognition is a challenging task that requires the integration of information from multiple modalities using advanced deep learning models and fusion techniques. The state-of-the-art systems in this field continue to evolve, with ongoing research focusing on improving the accuracy and robustness of these systems.

## DATASET

The initial dataset the project was based on was The Interactive Emotional Dyadic Motion Capture (IEMOCAP). This dataset is very popular for multimodal model training, and is used in the reference paper "Context-Dependent Domain Adversarial Neural Network for Multimodal Emotion Recognition". This dataset is not currently publicly available, after careful consideration the best possible alternative would be the Multimodal Emotion Lines Dataset (MELD) dataset.

The MELD is a publicly available dataset for multimodal emotion recognition research. It was introduced in the paper "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations" by Poria et al. (2018). The dataset consists of video clips from the popular TV show Friends, along with transcripts of the conversations, and annotations of emotional states.

Here are some academic aspects of the MELD dataset:

  * Size and Diversity: The MELD dataset contains over 13,000 utterances, making it one of the largest publicly available datasets for multimodal emotion recognition. The dataset includes diverse emotional states such as anger, sadness, joy, and disgust, along with neutral states.

  * Multimodal Data: The dataset contains multimodal data, including audio, video, and text modalities. The audio modality includes speech signals, while the video modality includes facial expressions and body language. The text modality includes transcripts of the conversations.

  * Annotations: The MELD dataset includes annotations of emotional states at the utterance level, along with annotations of the speaker's identity and gender. The emotional states are annotated using a four-point Likert scale, where 0 indicates a neutral state and 3 indicates a high-intensity emotional state.

  * Applications: The MELD dataset has been used in several research studies for multimodal emotion recognition, including studies that explore the use of deep learning models for this task. The dataset has also been used for applications such as sentiment analysis and affective computing.

  * Challenges: Despite its size and diversity, the MELD dataset has some limitations, such as its focus on a single TV show and its annotation scheme, which only includes a limited set of emotional states. However, ongoing research is addressing these limitations by developing new annotation schemes and collecting additional data from diverse sources.

## DATASET PREPROCESSING AND PREPARATION
Multimodal preprocessing for MELD typically involves several steps. First, the audio data must be transformed into a usable format, such as Mel-frequency cepstral coefficients (MFCCs), which are commonly used in speech recognition tasks. Finally, the text data must be preprocessed to remove stop words and perform tokenization, stemming, and other natural language processing techniques. Since we are not using the visual data for the initial implementation, we can exclude the preprocessing requirements for visual data.

Once the preprocessing is complete, the data can be aligned and merged into a multimodal representation for each dialogue. This representation can then be used to train machine learning models to predict emotion labels for each speaker. Multimodal preprocessing is a critical step in achieving high accuracy in emotion recognition tasks using the MELD dataset, as it enables the effective integration of multiple modalities to capture the complex nature of human emotional expression.

### Text
In order to preprocess the text, we start from scratch, preprcessing the utterances, creating the vocabulary and tokenizing the utterances. The steps are described below:

  * **Load the data**: This step involves reading in the data from the CSV file containing the MELD dataset. The CSV file contains the raw transcripts of the conversations, along with labels for various attributes such as the speaker ID, the emotion being expressed, and so on.

  * **Clean the data**: In this step, the raw transcripts are preprocessed to remove any unwanted characters or words, as well as any annotations or tags that are not needed for the task at hand. This can include removing special characters, punctuation marks, numbers, and other non-textual elements that may interfere with the analysis.

  * **Tokenize the data**: Once the raw text has been cleaned, it needs to be tokenized into individual words or tokens. This is done using the NLTK library's word_tokenize function, which splits the text into a list of individual words.

  * **Add speaker information**: In the MELD dataset, each utterance is associated with a specific speaker. To account for this, the speaker information is added to the tokens using special tags such as [A] or [B], which indicate the speaker ID for each utterance.

  * **Convert labels to numerical values**: The MELD dataset includes labels for various attributes such as the emotion being expressed, the sentiment of the utterance, and so on. These labels are converted to numerical values using a mapping function, which maps each label to a unique integer value.

  * **Split the data**: Finally, the preprocessed data is split into training, validation, and test sets, with a predefined ratio of data for each set. This ensures that the model is trained on a diverse range of data and can generalize well to new data.

### Speech
In order to obtain the resulting speech there was an initial investigation on the provided data, this investigation resulted in the following observations of problems within the dataset:

  * The train split contained a corrupted video file (dia125_utt3.mp4). As a workaround, we have duplicated the data from other similar utterances from the dataset of the same emotion (neutral).

  * In the test split, there are several files named as “final_videos_test*” that contain the audio corresponding to the annotations in the csv, so we have used their content for the training, replacing the files that follow the naming convention for the other splits. E.g. We used the audio data in final_videos_testdia108_utt0.mp4 instead of the data in dia108_utt0.mp4.

  * We found 8 video files containing audio of just a couple of hundred of milliseconds that weren’t usable for training and didn’t correspond to the annotations. These included 2 for the validation set (dia99_utt3.mp4 and dia108_utt5.mp4) and 6 for the training set (dia71_utt0.mp4, dia179_utt8.mp4, dia312_utt9.mp4, dia332_utt1.mp4, dia406_utt2.mp4 and dia523_utt10.mp4). The workaround was to use silence for these files and blacklist them in the training so that they weren’t skipped.

  * The number of audio channels in the source video files wasn’t consistent: depending on the utterance it contained 2 (stereo) or 6 channels (surround, 5.1). We decided to always use the first channel (left) for consistency in the data, though in the case of the surround files the center channel should have been preferred.

Once these fixes were applied, the initial data was cleaned and contained the correct information to be properly processed, we were able to generate a script (src/meldaudiopreprocess.py) that preprocess the data. The script expects the folder with the raw MELD dataset as input and an output folder where we expect the preprocessed data to be stored. This script runs through the following steps to preprocess the data:
  * Iterate over the source folders of the MELD dataset (train_splits, output_repeated_splits_test and dev_splits_complete extracted from the .tar archive) and creates the target folder(s) where to store the preprocessed data, if needed.

  * For each of the files in the split folder:
      * The special case for the files named “final_videos_test” is handled and the audio tensor is extracted from the .mp4 file with torchvision.io.read_video().
      The right channel is discarded and only the left channel is used to first carry out a downsampling from the source sample rate (48kHz) to the target sample rate, 8kHz. This way we want to speed up the preprocessing and the training in the neural network later, having speech information only up to 4kHz (following the Nyquist theorem).

  * After the resampling, we compute a log-MelSpectrogram of the audio using torchaudio.transforms.MelSpectrogram and taking torch.log afterwards. The window length we use is of 20 milliseconds (160 audio samples at 8kHz), a default hop size of 10 milliseconds (half of the window size), 1024 bins of FFT size and 40 mel bands. We also use the ‘center’ flag (by default) to pad the waveform on both sides so that each audio frame is centered. To avoid introducing -inf values into the neural network (which caused NaNs in the experiments), we replace these values with a silence constant of -100.0. Alternatively, this could have been worked around by adding an epsilon (very small value) to the raw audio beforehand. If the raw audio isn’t long enough to compute a MelSpectrogram, we store a tensor of 40 mel bands with the silence constant.

  * The values of the log-MelSpectrogram are then stored in a python dictionary (keys are the source .mp4 file and the value is a torch.tensor). The dictionary is then saved to a pickle file following the naming convention “meldaudiodata_{split_name}.pickle”.

## TEXT ARCHITECTURE
The following architectures were explored as part of the experimentation for emotion classification using textual representations.

### Simple Multilayer Perceptron (MLP) Classifier

The MLP Classifier model consists of the following layers:

  * **Embedding layer**: The Embedding layer maps the input tokens to dense vectors of embedding_dim dimensions. The weights of this layer are learned during training.

  * **Two fully connected layers**: hidden_size_1 and hidden_size_2 units, respectively. The MLP is used to learn non-linear feature representations from the input token embeddings.

  * **Linear classifier**: The Linear layer takes the output of the MLP and maps it to the n_emotions emotion classes.

During the forward pass, the input text tokens are first passed through the Embedding layer to obtain a tensor of shape (B, nº tokens/utterance, embedding_dim), where B is the batch size. The Embedding layer returns a tensor of the same shape, but with the embeddings of each token in the input sequence.

Then, the tensor is passed to a mean pooling operation, which averages the embeddings of each token along the sequence dimension (nº tokens/utterance), resulting in a tensor of shape (B, embedding_dim).

Next, the tensor is passed through the MLP, which learns a non-linear feature representation of the input. Finally, the output of the MLP is passed through the Linear classifier to obtain the final output of the model, which is a tensor of shape (B, n_emotions).

### Long Short Term Memory

The Long Short Term Memory Classifier consists of the following layers:
  * **Embedding layer**: This layer creates word embeddings for the input tokens. It takes as input the vocabulary size, embedding dimension, and padding index. It outputs the embedded representation of the input tokens. In the code you provided, the nn.Embedding module is used to implement this layer.

  * **LSTM layer**: This layer processes the input sequence of embeddings using a Long Short-Term Memory (LSTM) network. The LSTM layer takes as input the embedding dimension, number of layers, bidirectionality, batch size, and dropout probability. It outputs the hidden state and cell state of the LSTM network for each input token. In the code you provided, the nn.LSTM module is used to implement this layer.

  * **Linear classifier layer**: This layer maps the output of the LSTM layer to the predicted emotion labels. It takes as input the output of the LSTM layer, which is a concatenation of the final hidden states from both directions of the LSTM, and outputs a vector of size n_emotions, where n_emotions is the number of emotion labels. In the code you provided, the nn.Linear module is used to implement this layer.

Additionally, a dropout layer is used after the embedding layer and a mean pooling operation is performed on the output of the LSTM layer to obtain a fixed-length representation of the input sequence.

### Context Model
The context classifier model consists of the following layers:

  * **LSTM layer**: This layer is an instance of nn.LSTM and is responsible for processing the input sequence x through a bidirectional LSTM with two layers. The vec_dim parameter represents the dimension of the input and output vectors. The input shape of the layer is (nº utterance/dialogue, vec_dim) and the output shape is (nº utterance/dialogue, vec_dim), where nº utterance/dialogue refers to the number of utterances/dialogues in the batch.

  * **Dropout layer**: This layer is an instance of nn.Dropout and is applied to the output of the LSTM layer. The dropout parameter is set to 0.3, which means that 30% of the output elements are randomly set to zero during training to prevent overfitting.

  * **Linear classifier layer**: This layer is an instance of nn.Linear and is responsible for classifying the input sequence into one of seven emotion classes. The input shape of the layer is (nº utterance/dialogue, vec_dim) and the output shape is (nº utterance/dialogue, 7).

### Transformer

## SPEECH ARCHITECTURE

### 2D CNN Classifier
The 2D CNN Classifier model consists of the following layers:

  * **Instance Normalization**: This layer normalizes the input tensor along the channel dimension for each sample in the batch. It helps to ensure that the input to the model has a consistent range of values.

  * **Convolutional Layers**: These layers apply a set of convolutional filters to the input tensor to extract features from it. In this model, there are two sets of convolutional layers with a 3x3 kernel size and 20 output channels each. The output of the first convolutional layer has 20 channels and is passed through a max pooling layer to reduce the spatial dimensions of the tensor. The output of the second convolutional layer also has 20 channels and is again passed through a max pooling layer.

  * **Rectified Linear Activation Function**: This activation function is applied after each convolutional layer to introduce non-linearity into the model. It ensures that the output of the convolutional layers is positive.

  * **Dropout**: This layer randomly sets some activations to zero during training to prevent overfitting. It helps to ensure that the model doesn't just memorize the training data and is able to generalize to new data.

  * **Linear Layer**: This layer applies a linear transformation to the input tensor, mapping it to a tensor with num_classes output dimensions. It is used for the final classification of the input tensor.

  * **Global Average Pooling**: This layer takes the output tensor of the convolutional layers and calculates the average value of each channel across the spatial dimensions of the tensor. This results in a single feature vector for each input sample, which is then passed through the linear layer to produce the final classification.

### 1D CNN Classifier
The 1D CNN Classifier model consists of the following layers:
  * **Instance normalization layer**: This layer performs normalization on the input along the channel dimension.

  * **Dropout layer**: This layer randomly drops some of the connections to prevent overfitting.

  * **Convolutional layers**: This layer consists of two convolutional layers with kernel sizes of 3 and 128 and a stride of 1. The first layer has 4 times the number of filters as the number of input channels. The output of each convolutional layer is followed by a max pooling layer with a kernel size of 2 and a stride of 2. Both convolutional layers use the ReLU activation function.

  * **Linear layer**: This layer takes the output of the convolutional layers and performs a linear transformation to map it to the output size.

  * **Global Average Pooling**: This layer averages the output of the last convolutional layer along the time dimension.

### LSTM Classifier

This is the same classifier as that proposed in the text section, this classifier will be adapted to take in the processed audios with the correct size.

## MULTIMODAL ARCHITECTURE
In order to tie these two together, we propose a similar architecture to that used in the paper. We will train a model on the audio data and a different model on the text data, we will then exract the feature vectors of each utterance for both the text representations and the speech representations of the data. We will use the best models selected in the experimentation steps for both speech and text to extract the feature vectors. We will then train a model using the concatenated versions of these feature vectors.

## TEXT EXPERIMENTATION
Once our models have been decided, the next important step is to run experimentation over the resulting models with the preprocessed dataset. For these, the following constraints applied:
  * The dataset is divided in 9989 training utterances, 1109 validation utterances and 2610 test utterances. 
  * All models are trained with an Adam optimizer with learning rate 1e-3
  * The test batch size was chosen to always be 100

### Utterance-level classification
The intial proposal was to train the model using batches of random utterances. This form of training resolves the issue of having different pad lengths for different examples. Batching this data by similar length, we make sure that we minimize the padding requirements in our model. Training the model in an utterance level poses one main issue: it kills the context of the text, we will be sending together utterances pertaining to different episodes, seasons, etc. This will be further explored in the dialogue-level classisifcation.

#### MLP
The initial experimentation with the MLP served to test the relation between the embedding size and the output size of the linear layer. The model used to run the initial test had the following specifications:
  * Prueba 0:
    * Embedding: 50
    * Hidden: 25 
  * Prueba 1:
    * Embedding: 50
    * Hidden: 100 
  * Prueba 2:
    * Embedding: 50
    * Hidden: 100
    * Dropout: 0.5

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_mlp_1.png)

Two main conclusions came out of this initial experimentation, adding a dropout layer between the embeddings and the hidden layer. This dropout layer provides the regularization necessary for our model not to overfit to the training data (as seen in the figure above attempts 0 and 1 overfit to the training data). Furthermore, we decide to apply *hidden = 2***embedding* to all of the following tests. This stems from the fact that we obtain good results with an initial MLP model. The validation loss does not shown overfitting and reaches a best value of 1.30. And in relation with the test set, we get a weighted F1 score average of 0.53, a test loss of 1.22 and test acc of 0.58. As we keep testing this model and further models, we will see that these initial evaluation metrics are already up-to-par. Therefore, following this trend ensured good results.

With some initial testing, the next logical step was to research the best embedding dimension for the model. The next three tests ran with the following specifications:
  * Prueba 3:
    * Embedding: 16
    * Hidden: 32
    * Dropout: 0.5
  * Prueba 4:
    * Embedding: 128
    * Hidden: 256
    * Dropout: 0.5
  * Prueba 5:
    * Embedding: 256
    * Hidden: 1048
    * Dropout: 0.5
  * Prueba 6:
    * Embedding: 64
    * Hidden: 128
    * Dropout: 0.5

Prueba 3, Prueba 4 and Prueba 6 gave the most prominent results, therefore we will be using these as a comparison base. Prueba 5 was done with a very large embedding size and the model overfit very quickly to the training data.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_mlp_2.png)

As you can see in these graphs, for a small embedding size as emb=16, the model is slower at learning due to its diminished capacity. For a bigger capacity as emb=128, learning is faster but it gives more problems of overfitting. The best result in this trade-off of overfitting and learning will be choosing emb=64, although emb=128 does not show bad results we do see a slight trend to overfitting. There is not major progress surrounding the metrics. The best ones are obtained in prueba6 and are the same as in prueba2. Best val loss 1.32, weighted averaage of 0.53, test loss of 1.22 and accuracy of 0.58. From these tries we conclude that a embedding size of [50,150] and hidden=2*emb is obtains the best results.

The next logical line of research is to increase the size of the model, to initiate testing on a larger model we increased the number of hidden layers by one. The following are the specifications for the tests run with this new architecture:
  * Prueba 4:
    * Embedding: 128
    * Dropout: 0.5
    * Hidden 1: 256
  * Prueba 6:
    * Embedding: 64
    * Dropout: 0.5
    * Hidden 1: 128
  * Prueba 8:
    * Embedding: 64
    * Dropout: 0.5
    * Hidden 1: 128
    * Hidden 2: 128
  * Prueba 9:
    * Embedding: 64
    * Dropout: 0.5
    * Hidden 1: 256
    * Hidden 2: 128
  * Prueba 10:
    * Embedding: 64
    * Dropout: 0.5
    * Hidden 1: 256
    * Dropout: 0.5
    * Hidden 2: 128
  * Prueba 11:
    * Embedding: 64
    * Dropout: 0.5
    * Hidden 1: 256
    * Dropout: 0.5
    * Hidden 2: 512

Although many tests were run, for the purpose of comparison we have decided to extract the best between both architectures to be able to extract conclusions from the model.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_mlp_3.png)

Looking at both graphs, the main conclusion is that there is not a clear advantage of making our model with one more layer. What is curious about the examples shown in the graph, is that the metrics in prueba8 are a little bit better. We reach a weighted average of 0.55, test loss of 1.21 and accuracy of 0.6. Although not a huge increase, these are the slightly better metrics obtained with an MLP model.

These is an interesting point to keep talking about. The best results we get on validation loss are around 1.32, at best 1.30. After that, overfitting starts and the training loss we stop around 1.10-1.20. What its interesting is that the results of test loss we are getting are around 1.22 at best, much better that validation loss. This is because the test set is almost of double length than the validation set. So obviously, overfitting is also going to affect the test set, but a little bit later than in the validation set and the best point obviously is going to be better. 

#### LSTM
LSTMs contain the context of the utterances within the model, therefore we hope that this is sufficient to extract better results from the model. Our first experiments are with one layer of unidirectional LSTMs and changing the embedding size. These specifications are shown below:
  * Prueba 0:
    * Embedding: 64
    * LSTM: 64
  * Prueba 1:
    * Embedding: 64
    * LSTM: 64
  * Prueba 2:
    * Embedding: 128
    * LSTM: 128
  * Prueba 3:
    * Embedding: 32
    * LSTM: 32

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_1.png)

Looking at the validation loss we see that the best result in terms of overfitting is prueba3, and also that prueba2, with emb=128, gives a kind of overfitting that we didn't see in MLP when we put emb=128. This is a sign that a layer of LSTM is more powerful, and therefore more keen to overfit, than the MLP we tried. 

Talking more about prueba3, we see that the validation loss gets to the typical results we have been getting and also the metrics are already quite good (weighted average = 0.54, test loss = 1.22, test accuracy = 0.59). 

The next step we think of is putting a bidirectional layer. By making the LSTM bidirectional it will be able to obtain context from future words in the text, as well as previous words, this means that we expect the new architecture to somewhat improve the results. The following are the specifications for the bidirectional tests:
  * Prueba 4:
    * Embedding: 64
    * Bidirectional LSTM: 64
  * Prueba 5:
    * Embedding: 32
    * Bidirectional LSTM: 32
  * Prueba 6:
    * Embedding: 16
    * Bidirectional LSTM: 16

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_2.png)

The graphs confirm what was previously commented about the power of LSTMs, because it overfits much faster with emb=64. Also comparing prueba3 and prueba5, we see that putting a bidirectional layer makes the training much better, with a better validation loss. Talking about the metrics, the results also get a little bit better as test loss reaches 1.2 wit the same accuracy 0.59.

Now is the moment to try if it gets better with more layers of bidirectional LSTM. The following specifications contain the tests run while increasing the architecture:
  * Prueba 7:
    * Embedding: 32
    * Bidirectional LSTM: 32, num_layers=2
  * Prueba 8:
    * Embedding: 32
    * Bidirectional LSTM: 32, num_layers=4

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_3.png)

The validation loss curve show that there is not many differences between them but it seems that prueba8, with 4 Bi-LSTM, starts getting worse results due to overfitting. It also seems that with 2 Bi-LSTM there is a little better behaviour in relation to overfitting. This is one of the reasons why we will continue using 2 Bi-LSTM. The other reason is that when added a little more dropout to the LSTM layer, as in prueba10, we get a weighted average of 0.56 (best until now), a test accuracy of 0.6 and a test loss of 1.19, the first time that we see it go under 1.20.
  * Prueba 10:
    * Embedding: 32
    * Bidirectional LSTM: 32, num_layers=2, dropout=0.3

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_3.png)

The last idea we had to try to improve the architecture is to add a little MLP after the LSTM. This tests was run with the following characteristics:
  * Prueba 12:
    * Embedding: 32
    * Bidirectional LSTM: 32, num_layers=2, dropout=0.3
    * Linear: 128

In the following graphs we see that it does not make any significant change, and looking at the metrics, we get the same test loss of 1.19 and accuracy of 0.6.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_5.png)

The last two tries worth mentioning are the following. Until now, we were updating with the LSTM the representations of each word. As the batches are of variable length, we need to somehow fix dimensions before sending the vectors to the classifier. What we have used throughout the work is to do the mean of all the vectors of the utterance. At this point, we tried to change it and used the maximum (at each component of the vectors: maximum component 0 of all vectors, maximum component 1...) The results are basically the same, but it is worth mentioning.

The other idea is that, as previously mentioned, we were changing the representations of the words with the LSTM. But you can also take the last representation of an LSTM, which should have all the meaning of a sentence. The following is a comparison between the best result of our previous approach (prueba10) and a result obtained extrapolating the same architecture with the approach just 
expalined (prueba16). The graphs below show the previous approach may be slightly better, although they are similar. The metrics are quite similar.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_lstm_6.png)

So this would be the end of the experiments with LSTM. As a recap of MLP and LSTM, both of them get stuck in the same values of val loss (at best 1.30), weighted average (0.56), test loss (around 1.20) and accuracy (around 0.6). It is true that the better metric results tend to be with LSTM, but the difference is very slightly.

#### Transformers

According to the literature, transformers and attention performance should be great, as they were vital in the improvements of results in the NLP field. We will start our experiments with a configuration of embedding size 128 and we will try different numbers of encoders and of heads in the multiattention block. What we will never change is that the dimension of the feedforward block is 4*embedding.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_transformers_1.png)

We can see that it is overfitting since the first moment. That is not surprising as these models are huge and the power of overfitting is very high. A dropout helps to see that this configuration works.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_transformers_2.png)

The best value of validation loss we get is 1.35. Metric are not bad as we get a test loss of 1.25 and accuracy of 0.58. But there should be room for improvement. 

We now want to check how the results change adding more encoders. The results are not good, the model does not learn at all. The loss increases since epoch 1 as in the following example of 4 heads and 4 encoders (prueba1). But the same happens in other combinations as 8 heads 4 encoders or 8 heads 8 encoders.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_transformers_3.png)

This makes us continue in the path of 2 encoders and 8 heads, which were the best results before. Using a lesson of the past results, where we see that using a smaller embedding dimension felt good to a powerful architecture as LSTM, we now try to reduce the embedding size previously used.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_transformers_4.png)

We can see that the results on the validation loss are similar, quite noisy and with a best val loss of 1.35. On the metrics, it improves a little bit to a test loss of 1.22 and a accuracy of 0.6, but continue to be results that we also get with MLP and LSTM. 

As there is not many more parameters to change, the results of the transformers are similar or even a bit worse than the ones got with MLP and LSTM. A reason we find is that the size of the dataset (train set of less tha 10000 utterances) does not let to exploit the advantage of transformers.

### Dialogue-level classification

Although the hyperparameter space is very big and maybe we have missed a different architecture that gives better results, it really seems that with one model, we reach the best results we can get with easy to implement architectures. This feeling is confirmed looking at the benchmark results of the MELD paper (2017), where text models reach similar F-scores and accuracys. But looking at how we did the training, we did batches of random utterances of similar length so that we had to include the less padding as possible. Now, we can have for each of these sentences a feature vector, with the same dimension for each sentence. Let's exploit this characteristic to do a second training of a second model. What we want of this new model is to take as inputs the feature vectors of the utterances of a dialogue. So we will pass dialogue by dialogue to this new model and we expect to change a little bit the representation of each utterance based on the context of the dialogue, based on the influence of the other utterances. 

The idea is to extract the feature vectors from the first model, when it is correctly optimized through training. Then we reorder all the vectors so that they are correctly sorted in their dialogue. Lastly, we send, dialogue by dialogue, the new data to the second model and try to optimize it. 

As the second model is a "context-model", we decided that a good choice would be to implement an LSTM architecture. For the first model we will start using the best results from the MLP/LSTM models in the previous section, and maybe modify them a little bit. As a lot of the discussion has been made before, the tests in this section will be based on the conclusions extracted in the previous section.

"The last detail to mention before starting the experiments is that now we tried to change from batch size 10 to 100 in the first model. This should make it a little bit more robust to outliers, but in the practice the results are the same, only that we need more epochs to run." 

Let's start the experiments applying a MLP as the first model. It will be the one with the best result we got, which to remember is a embedding size of 64 and two hidden layers with 128 neurons each. Then the second model, also taking the best structure found, is a 2 layer LSTM. At first we don't try it bidirectional because we think a future utterance should influence the emotion of a previous utterance. But later we will attempt to change this. 

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_1.png)

Now, let's show the performance of both models in this first experiment. The graph of the training loss makes a lot of sense. We could say that the data of the "model0" is the "raw data" so at first the model needs to learn the embeddings of each word, start to learn the relations of them with the  emotions... but the "model1" has as data optimized feature vectors that already have a lot of information condensed, so the loss of the second model starts much lower although its parameters are randomly initialized. But talking about the validation loss we see that in model0 behaves well but finish in clear overfitting, and in model1 starts already in overfitting and there is not 
improvement.

Now we will show another try in which now is the moment we changed from batch size 10 to 100, therefore we increase a little bit the epochs we run, and also add a dropout at the start of the second model. 

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_2.png)

The main difference in the validation loss in comparison with the previous graphs is that at the end of the model0 optimization, is not in a clear overfitting situation. Therefore, we give model1 a good starting point which, thanks to the context of the dialogue, is able to break the barrier we had of val loss being always greater than 1.30. In this simulation we can see it goes down 1.30 and 
reaches 1.27 in epoch 5.

Talking about the metrics, both model0 and model1 have a test loss of 1.22 and the accruacy of model0 is 0.59 while in model1 is 0.60. So they are very similar but obviously, the point is to try to improve them or at least remain the same, when adding model1.

We will continue testing this approach but we have already seen its power. When simulating model0 to its optimization point (not overfitting), adding the dialogue context with model1 helps to generalize the results and decrease a little bit more the validation loss in comparison to model0, breaking a barrier we could never break with our previous models without dialogue context. In our previous experiments, MLP results were not affected at all if we put 2 or 3 layers and LSTM worked a little bit better if its dimension was a little bit smaller than 128. Therefore, let's try now and MLP with embedding size of 32 and a hidden of 64, and so the LSTM entering dimension will also be of 64.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_3.png)

In model0 we could run for 40 epochs until a sign of flattening could be seen. Model1 overfits more so that is why we stop in epoch 10. But the results continue to be the expected. We can see model0 is unable to go below 1.30 in 40 epochs, but when added the dialogue context, the validation loss drops and reaches now 1.25, the best result found. Although this does not transalate to the metrics (test loss = 1.22 for both and accuracy of 0.57 and 0.58 respectively), this result means that the model is learning how to generalize a little bit more thanks to the context.

Our last try in this approach is going to use two LSTM, one for each model. The reason is because we decided at the end of the first part of the discussion that LSTM approach was the one which gave better metrics with also a good validation loss (go back Table and graph comparing MLP/LSTM/Transformer). So we are going to build a model0 with embedding size of 32 and a 2 layer 
bidirectional LSTM, and a context model1 with a 2 layer LSTM (no bidirectional) of input size 64.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_4.png)

Results are interesting. Val loss of model1 is below the one of model0, reaching in a point 1.28 but on 1.30 on average. Visually the behaviour is a little bit worse than in previous experiments. But, both model0 and model1 metrics are better. Both of them have a test loss of 1.18 but an accuracy of 0.6 (model0) and 0.62 (model1). This last value is the best test accuracy obtained until now. 

Also, the weighted average of model1 is 0.57, the best result we got. So the behaviour of the 2 LSTM's is similar as before, we don't see any advantage in terms of validation loss in comparison to MLP+LSTM, but it gives back a little improvement in the test metrics. Only a quick mention. In model1 we also triedd a 2 layer bidirectional LSTM, but results does not change at all.

Now we make a quick explanation of another way we tackle the problem of dialogue context. In this approach we train both models at the same time, hoping model1 learns more things than before. The idea is to run the training simultaneously. We reun 1 epoch of model0 and update its parameters, but we also take the feature vectors and send it to model1, run also 1 epoch and update its parameters. In the first epochs the feature vectors will not be very good but they will get better. 

The experiment we run of this approach is a model0 of embedding size 64 and hidden 128 with a 2 layer LSTM with input size of 128. At first we choose batch size of 100 and 15 epochs.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_5.png)

We see on the graph the training and validation loss. Taking a look more to the validation, the validation loss of model1 is always better than the one of model0, which makes sense. Also, as there is not overfitting in model0, the best results of both validations are in the last epoch, which is logical also. According to the metrics, model0 has a test loss of 1.23 and accuracy of 0.58, while model1 has a test loss of 1.21 and accuracy of 0.6. So everything makes sense. But we are going to do the same experiment for more epochs, because val loss in model0 is 1.34 (and val loss is 1.29 in model1) and we know it can get lower. As there is not overfitting yet, we can try to do more epochs. 

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_6.png)

In this graph we show now 30 epochs. Val loss in model0 is now around 1.32 and getting flat in the last epochs. But thanks to this decrease, val loss in model1 reaches now 1.27. Also, the improvement in the metrics continue, having model0 a test loss of 1.21 and accuracy of 0.59 and model1 a test loss of 1.2 and accuracy of 0.61.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/text_dialogue_7.png)

In this last try we run for 50 epochs. We now see the overfitting and that the epoch in which the validation loss has a minimum in model0 is the same epoch of the minimum of model1. The behaviour is the same for both validation losses, but the one of model1 is always below. This basically is another proof of the help of the dialogue context in the learning of our model.

So the conclusion after these tries is that the dialogue context obviously helps reducing the validation loss and that the optimum point of this approach is when model0 reaches the minimum of validation loss, that if your remember is what we did before. Optimize model0, get the feature vectors and optimize the context model, model1.

## SPEECH EXPERIMENTATION
Once our models have been decided, the next important step is to run experimentation over the resulting models with the preprocessed dataset. For these, the following constraints applied:
  * Each utterance has a vector of dimention 1611.
  * The dataset is divided in 9989 training utterances, 1109 validation utterances and 2610 test utterances. 
  * All models are trained with an Adam optimizer with varying learning rates. This is because we attempted to use the same learning rate for all tests but with the previously proposed dataset the model did not learn properly.
  * The test batch size was chosen to always be 100.

### Utterance-level Speech Classification

For our base model we will use the model proposed by MELD in github. This is an LSTM with the following hyperparameters:
  * epochs: 30
  * lr: 1e-4
  * batch: 50

The results obtained are portrayed in the graph below. 

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/speech_lstm_1.png)

Fruther testing with different parameters was required.

**Test 2:**
  * epochs: 30
  * lr: 1e-5
  * batch: 50

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/speech_lstm_2.png)

As is expected, the model grossley underperforms compared to the previous results seen in the text experimentation. It is clear that it is harder to extract relevant information from speech than from text. This is reflected through worse metrics in speech compared to those in text; higher loss, lower accuracies, high amounts of epochs required to reflect training on the model, etc.

### Dialogue-level Speech Classification

The same way as we trained the initial models in the text classification, the previous models were tested in utterance level batchings that were randomized. In order to test the importance of context we will use the same context model exposed in the text section and will compare that shown in the results shown in the utterance-level speech classification section with those extracted from this model.

![alt text](https://github.com/UPC-AIDL-MER/multimodal_emotion_recognition/blob/main/images/speech_dialogue_1.png)

From the graph shown above, we can see that there is no significant improvement in the minimum validation loss from using context. Although we do see a decrease a small decrease of the validation loss, it is not enough to compare this to the results obtained from experimentation with the textual data. This highlights the difficulty of extracting information relvant to emotion classification from speech.

## MUTLIMODAL EXPERIMENTATION
The multimodal experimentation was done by running a model with the feature vectors extracted from the individual speech/text models. An initial model was trained on concatenated feature vectors with the following characteristics:
  * 

## CONLCUSIONS

The results from the initial paper are exposed below:


The results from our experimentation are exposed below:


## FUTURE WORK
  * **Data Augmentation**: An initial idea in order to test the functioning of our model in different situations, was generate more data from the pre-existing MELD dataset. This was specifically thought out for the speech section by adding noise to the spectrograms. Using this augmented data to train the models would have given us more room to work with the speech model and potentially have impacted the metrics of the model.
  * **Audio Preporcessing and Usage**: Currently there are many techniques being used to extract usable data from audios, given more time we would have been able to play around with different audio processing techniques to be able to find which was more efficient. As mentioned in the previous sections 1D and 2D CNN Classifiers were not capable of learning with the data that we had, we could have further experimented with techniques to make these properlly work and extract useable data. Other ideas we could have experimented with wav2vec to generate features, amongst others.
  * **Add visual data**: In order to reduce the complexity of our model we opted to add text and speech data rather than include all three aspects of the mutlimodal capacity of the data. In this way, we could make our model better by adding the facial queues and movement queues incluedd in the video of the tv show. Although this might not provide as much information as the textual data (in the same form that the speech model was not able to reach the same results as the text model) it adds context so our multimodal model is able to properly contextualize the situation.
  * **Experiment with different types of concatenation**: For the purpose of multimodal experimentation we have used direct concatention of the audio and speech feature vectors. We could experiment with different ways to concatenate these and what effect they have in the training process of the model.