

import subprocess
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import os
import speech_recognition as sr
from pydub import AudioSegment
import ffmpeg
import stanza
from googletrans import Translator
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain.prompts import PromptTemplate
import pandas as pd
import re

import warnings
warnings.simplefilter('ignore')

"""### Scene Detect"""


def split_video (video_path):
    print("Scenes Detecting")
    # scene_list = detect(video_path,AdaptiveDetector())
    # split_video_ffmpeg(video_path, scene_list)
    # NUM_SCENES = len(scene_list)

    output_directory = "/content/drive/MyDrive/video_seg/framesxxxxx"
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)
    command = [
    'scenedetect',
    '--input',video_path ,
    'detect-adaptive', 'list-scenes', 'save-images', '-n', str(1),'--output', output_directory]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    scenes = pd.read_csv(video_path.split(".")[0] +"-Scenes.csv")
    scenes = scenes.reset_index()
    scenes.columns = scenes.iloc[0]
    scenes.drop(0,  inplace = True)
    scenes = scenes.reset_index(drop=True)
    scenes = scenes.drop(scenes.columns[-3:], axis=1)
    scenes.to_csv("uploaded_video-Scenes.csv", index=False)
    

def convert_mp4_to_wav(mp4_file_path):
    path_no_extention = mp4_file_path.split(".")[0]
    mp4_2_mp3 = "ffmpeg -i "+ path_no_extention + ".mp4 "+ path_no_extention +".mp3"
    mp3_2_wav = "ffmpeg -i " + path_no_extention + ".mp3 " + path_no_extention +".wav"
    os.system(mp4_2_mp3)
    os.system(mp3_2_wav)

def speech_to_text(audio, scene_num):
    extracted_text = []
    recognizer = sr.Recognizer()
   #recognizer.energy_threshold =300
    recognizer.dynamic_energy_threshold = True

    if (len(audio) < 5* 1000 ):  #less than 5 sec,
        extracted_text.append("no text")
        print("small scene")

    elif((len(audio) < 200* 1000)):  # the recognizer could detacet it direct
        print("middel scene")
        audio.export(f'scene_{scene_num}.wav', format='wav')
        with sr.AudioFile(f'scene_{scene_num}.wav') as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_segment = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_segment, language='ar-AR')
                extracted_text.append(text)
            except sr.UnknownValueError:
                print(f"Could not understand scene_{scene_num +1}")
                extracted_text.append("no text")
            except sr.RequestError as e:
                print(f"Error with the recognition service; {e} in scene_{scene_num + 1}")
                extracted_text.append("error")
    else: # the scene is too long
        print("big scene")
        # Set the duration for each segment in milliseconds
        segment_duration = 180* 1000
        segments = [audio[i:min(i+segment_duration,len(audio) )] for i in range(0, len(audio), segment_duration)]
        for i, segment in enumerate(segments):
          segment.export(f'segment{i}.wav', format='wav')
          with sr.AudioFile(f'segment{i}.wav') as source:
              recognizer.adjust_for_ambient_noise(source)
              audio_segment = recognizer.record(source)
              try:
                  text = recognizer.recognize_google(audio_segment, language='ar-AR')  # Adjust language if needed
                  extracted_text.append(text)
              except sr.UnknownValueError:
                  print(f"Could not understand segment {i} in  scene_{scene_num}")
              except sr.RequestError as e:
                  print(f"Error with the recognition service; {e} in scene_{scene_num}")

    if(len(extracted_text)==0): # this case when the scene is long and no text extracted from  all parts
        return "no text"
    else:
        return ' '.join(extracted_text)



def text_per_scene(NUM_SCENES, scene_end_time ,audio_path):
  # end time in sec
    text = []
    audio = AudioSegment.from_wav(audio_path)
    for scene_num in range ( NUM_SCENES):
        if(scene_num == 0):

            audio_scene = audio[0: float(scene_end_time[scene_num]) * 1000]
            text.append(speech_to_text(audio_scene,scene_num) )
        else:
            audio_scene = audio[float(scene_end_time[scene_num - 1]) * 1000: float(scene_end_time[scene_num]) * 1000]
            text.append(speech_to_text(audio_scene, scene_num))
    return text


def analyze_text(text):
    nlp = stanza.Pipeline(lang='ar', processors='tokenize,ner')
    doc = nlp(text)
    entities = [{'كيان': ent.text, 'نوع': ent.type} for ent in doc.ents]
    return entities

translator = Translator()

def translate_from_arbic(text):
    return translator.translate(text, dest='en').text

def translate_from_english(text):
    return translator.translate(text, dest='ar').text

def LLama2_loading():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm=HuggingFacePipeline(pipeline=pipeline,pipeline_kwargs={"max_new_tokens": 80},model_kwargs={'temperature':0.7})
    return llm

def creat_title_chain(llm):
    prompt = PromptTemplate(
        template="""You are a helpful assistant, could you put a title for this {text} the title must form of five words maximum


        title:
        """,
        input_variables=["text"]
    )
    chain = prompt | llm
    return chain


def create_scene_title(video_path):
    #NUM_SCENES, scene_end_time = split_video(video_path)
    # print(NUM_SCENES, scene_end_time)
    
    scenes = pd.read_csv(video_path.split(".")[0] +"-Scenes.csv")
    NUM_SCENES = len(scenes)
    scene_end_time = scenes["End Time (seconds)"]
    Video_name = video_path.split(".")[0]
    convert_mp4_to_wav(video_path)
    print("extracting text from scenes .....")
    text = text_per_scene(NUM_SCENES,scene_end_time ,Video_name+".wav" )
    print("extract text from scenes done .....", '\n')
    #print(text)
    llm = LLama2_loading()
    print("creating title....")
    chain = creat_title_chain(llm)
    #entities = []
    title_list = []
    english_pattern = re.compile(r'[a-zA-Z]')
    for sample_text in text:
        if(sample_text == "no text"):
          title_list.append("لا يوجد نص")
        else:
          eng_text = translate_from_arbic(sample_text).replace("\n", "")
          title = chain.invoke({"text": eng_text})
          arb_title =  translate_from_english(title).replace("\n", "")
          
          title_list.append(re.sub(english_pattern, '', arb_title))
          #entities.append(analyze_text(sample_text))
    print("creating title done....")
    title_df = pd.DataFrame.from_dict({"Title": title_list,"Text":text })
    title_df.to_csv('text_title.csv', index=False)
    



