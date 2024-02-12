# Generative_Video_Indexing
 
## Problem definition:
The problem is to detect scenes in a video and create a title for every scene in the video, to help to search for specific parts in the video.
The title must have all information not only speech information but also scene details and background.
## Demo


https://github.com/doaa-sala7/Generative_Video_Indexing/assets/61519327/f3ef7d14-50f5-4c98-900b-1cdfd6023e05


## Data: 
* Input Data: Arabic video
* Output Data: CSV file has scene details (start time, end time, number of frames, scene Title)
## Methodology:
It is divided into three parts:
* Detect scenes
* NLP approach
* Vision approach

### Detect scenes 
* Detect scenes using the PySceneDetect library
* It has three detection methods:
1.	Threshold scene detection (detect-threshold): analyzes the video for changes in average frame intensity/brightness
2.	Content-aware scene detection (detect-content): based on changes between frames in the HSV color space to find fast cuts
3.	Adaptive content scene detection (detect-adaptive): based on detect-content, handles fast camera movement better by comparing neighboring frames in a rolling window

* In this project we used (detect-adaptive) in the Detection
* Took one keyframe per scene
  
### NLP approach:
1.	Convert Video to wav in function (convert_mp4_to_wav)
2.	Extract Audio per scene:
In the method (text_per_scene), we have the scene details from the detection method, so using the end time of every scene to extract specific audio to send it to (speech_to_text) method
3.	Speech to Text
We are using Google Recognizer to convert speech to text, but it has some limitations, it can’t convert a large audio, so if the audio is larger than 200 seconds, we segment it into small audios.
The second limitation is the ambient noise, the recognizer can’t detect the speech when the ambient noise the high, to solve it we used (adjust_for_ambient_noise) method for every scene, and made the energy level dynamic, the results were enhanced but not 100 percent.
We also consider the scene doesn’t have speech if it is smaller than 5 second
4.	Translate from Arabic to English
We Translated the text from Arabic to English to use llama2
5.	Create a Title chain
We create a Prompt Template to create a title for every scene and create a langchain using llama2
6.	Translate from English to Arabic
We translate the English Title which is the output of the Title chain to the Arabic Title
7.	Create a CSV File with These results

### Vision approach
1.	Then we take the frames from each scene passing them to the the DataSet class in the dataset.py script to handle batches of the data taken  
2.	The dataset.py also includes functions for preparing and preprocessing the train data and the eval data 
3.	In the base_model.py and the model.py I Leveraged a fine-tuned CNN, like ResNet50, for robust image feature extraction.
4.	Process features with an LSTM network to capture sequential patterns, enhanced by an attention mechanism for focused image understanding.
5.	Apply attention to highlight relevant image regions during each captioning step.
6.	Convert LSTM outputs into vocabulary logits to predict subsequent caption words.
7.	The image_caption/models folder has 2 models one was purely trained only on the COCO2017 dataset and has very good generalism and accuracy the other which is used for demo purposes in this project was trained on a custom-made dataset
8.	The optimization  and loss measures techniques are also mentioned in detail in the presentation 
9.	The results of the model are then applied to the translate function using pandas and the English text resulting from the model is turned into Arabic and then concatenated with the title captions from the Llama2 model in such a way that each speech-to-text title and each visual-to-text caption is aligned within the same timestamp/time frame it happened on, look for the csv metadata results we produced for more clarity 

