# Generative Video Indexing
 
## Overview:
The objective of this project is to develop a system capable of automatically detecting scenes within a video and generating descriptive titles for each scene. This indexing process aims to facilitate efficient searching and navigation within the video content by providing comprehensive titles that encompass not only speech information but also scene details and background context. By accurately labeling scenes with informative titles, users can quickly identify and locate specific parts of the video content based on their content and context.
## Table of Contents
* Introduction
* Demo
* Data
* Workflow And Methodology
* Results
* Contributors
## Introduction
The primary objective of this project is to automatically identify distinct scenes within a video and create informative titles for each scene. Traditional video indexing methods often rely solely on speech recognition or basic scene detection techniques, which may overlook crucial visual cues and context. Therefore, the goal here is to develop a system capable of generating descriptive titles that encompass all relevant information within a scene, aiding users in locating specific content within the video.
## Demo


https://github.com/doaa-sala7/Generative_Video_Indexing/assets/61519327/f3ef7d14-50f5-4c98-900b-1cdfd6023e05


## Data: 
* Input Data: Arabic video
* Output Data: CSV file has scene details (start time, end time, number of frames, scene Title)
## WorkFlow and Methodology:
![video_indexing drawio](https://github.com/doaa-sala7/Generative_Video_Indexing/assets/61519327/8102e691-93a4-4e02-99b7-44f78904d675)

It is divided into three parts:
* Detect scenes
* NLP approach
* Vision approach

### Detect scenes 
* The first step involves employing advanced computer vision techniques to detect scene transitions and segment the video into distinct scenes.
* Detect scenes using the [PySceneDetect library](https://www.scenedetect.com/)
* detect-adaptive fanction is used in the Detection, and one keyframe is taken per scene.
### Title Generation:
Once scenes are identified, the next step is to generate descriptive titles for each scene. This task requires a combination of natural language processing (NLP) and computer vision techniques to extract pertinent information from the scene, including dialogue, visual elements, and contextual details. 

## Results
* the Result is a CSV file for all scenes Details and Arbic Title based on the speech and image captioning based on the scene.

## Contributors
* Doaa: Scene detection and NLP approach
* Omar: Vision approach


