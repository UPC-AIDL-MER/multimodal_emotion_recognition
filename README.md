# Multimodal Emotion Recognition
Final project for the 2022-2023 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Álvaro García Hernández**, **Carla Campàs Gené**, **Hector Colado** and **Jorge Garcia**. 

Advised by **Carlos Escolano**.


Table of Contents
=================

  * [INTRODUCTION AND MOTIVATION](#introduction-and-motivation)
  * [BACKGROUND](#background)
  * [DATASET](#dataset)
  * [DATASET PREPROCESSING AND PREPARATION](#dataset-processing)
    * [Text](#nl-data-processing)
    * [Speech](#speech-dataset-processing)
  * [TEXT ARCHITECTURE](#text-architecture)
	 * [Simple MLP Classifier](#optical-flow)
	 * [RNN Model](#rnn-model)
  * [SPEECH ARCHITECTURE](#speech-architecture)
  * [MULTIMODAL ARCHITECTURE](#multimodal-architecture)
  * [EXPRIMENTATION](#experimentation)
     * [MODEL IMRPVEMENTS](#model-improvements)
     * [HYPERPARAMETER TUNING](#hyperparameter-tuning)
  * [FUTURE WORK](#future-work)

## INTRODUCTION AND MOTIVATION

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

### Text

### Speech

## TEXT ARCHITECTURE

### Simple MLP Classifier

### RNN Model

## SPEECH ARCHITECTURE

## MULTIMODAL ARCHITECTURE

## EXPERIMENTATION

## FUTURE WORK