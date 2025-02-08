## Overview

This project is a video content generation application that leverages Google Cloud's Vertex AI and GloVe embeddings to create videos based on user input. Users can generate videos from keywords, custom prompts, or uploaded images. The application uses Streamlit for the user interface and integrates various machine learning models for image generation and text processing.

## Features

- **Generate Video from Keywords:** Users can select keywords to create a video.
- **Custom Video Generation:** Users can describe what they want to create in their own words.
- **Image-based Context Generation:** Users can upload images to provide context for video generation.

## Requirements

To run this project, you need the following dependencies:

- Python 3.9 or higher
- Streamlit
- Gensim
- FAISS
- NumPy
- Google Cloud Vision API
- Langchain
- MoviePy
- Pillow

### Install Dependencies

## Note
1. The embedding uses glove library that can be downloaded from https://github.com/stanfordnlp/GloVe/tree/master/src

  2.The program uses Vertex AI and one needs to genrate the api key for Vertex AI from GCP and add key.json file in the repository.

3. to run it in local use streamlit run main_user.py
##
