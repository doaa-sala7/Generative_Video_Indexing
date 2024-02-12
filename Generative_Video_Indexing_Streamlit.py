import streamlit as st
import pandas as pd
import os
from Speech_Title_optimized import LLama2_loading
from Speech_Title_optimized import create_scene_title
from Speech_Title_optimized import split_video
from image_caption import creat_caption
from Speech_Title import translate_from_english

def continue_creating(destination_path):

    
    create_scene_title(destination_path)
    caption = pd.read_csv("captions.csv")

    scenes = pd.read_csv("uploaded_video-Scenes.csv")
    title = pd.read_csv("text_title.csv")

    scenes["img_caption"] = caption["caption"]
    scenes ["text_title"] = title["Title"]
    scenes['img_caption'] = scenes['img_caption'].apply(translate_from_english)
    scenes.to_csv('final_results.csv', index=False)


def save_uploaded_video(video_file, destination_path):
    try:
        # Save the uploaded video to the specified destination path
        with open(destination_path, 'wb') as f:
            f.write(video_file.read())
        #st.success(f"Video saved successfully to {destination_path}")
    except Exception as e:
        st.error(f"Error saving video: {e}")


# Use Markdown with HTML to center the title
st.markdown("<h1 style='text-align: center;'>Generative Video Indexing</h1>", unsafe_allow_html=True)

with st.sidebar.container():
    st.image("Generative Video Indexing.png", width=10, use_column_width=True)



# Flag to check if a video has been uploaded
video_uploaded = False
destination_path = 'uploaded_video.mp4'
# Video upload from file
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    
    # Display output video
    #st.video(uploaded_video)
    video_uploaded = True
    #st.write(f"Uploaded video: {uploaded_video.name}")  

    # Save the video to Google Drive
    save_uploaded_video(uploaded_video, destination_path)

# video_results_path = "/home/user/Downloads/amr.mp4"  # Replace with your video path

csv_file_path = "final_results.csv"  # Replace with your CSV file path

st.button('continue creating', on_click=continue_creating, args=[destination_path])

if  os.path.exists(destination_path):
  st.success(f"uploading video successfully Done")
  st.text("Video processing:")
  st.text("create caption")
  st.text("first step of Processing Video Done:")
  
  #st.write(f"uploading video Done ")

# Display the video with the analysis results
if video_uploaded:
    #video_path = os.path.dirname(uploaded_video.name)
    #st.write("Directory of uploaded video:", video_path)
    st.text("Video processing:")
    split_video(destination_path)
    st.text("create caption:")
    creat_caption()
    #create_scene_title(destination_path,llama2 )
    st.text("first step of Processing Video Done:")

    #st.video(video_results_path)

    # Check if the CSV file exists and then display it
if os.path.exists(csv_file_path):
    try:
        # Load CSV file
        dataframe = pd.read_csv(csv_file_path)
        # Display DataFrame
        st.write(dataframe)
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
else:
    st.warning("Waiting for CSV file to be generated...")





