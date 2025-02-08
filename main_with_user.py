import streamlit as st
import os
import vertexai
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from vertexai.preview.vision_models import ImageGenerationModel
from moviepy import *
import numpy as np
from PIL import Image
import io
from user import UserManager
from rag_processor import RAGProcessor
from gensim.models import KeyedVectors
from user import UserManager
from google.auth.transport.requests import Request
#from credentials import credentials
#credentials.refresh(Request())
#glove_path = "glove.6B.300d.txt"
# Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
PROJECT_ID = "pure-vehicle-448919-g7"
REGION = "southamerica-east1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Set up models
llm = VertexAI(
    model_name="gemini-1.5-pro-001",
    max_output_tokens=256,
    temperature=0.5,
    top_p=0.8,
    top_k=40,
    verbose=True,
)
image_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

# Initialize UserManager
user_manager = UserManager()

# LLM Chains
# add_context = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate(
#         input_variables=['user_prompt'],
#         template="Based on this description: {user_prompt}, suggest a few more details and explain it. Don't include a frame on the image."
#     ),
#     output_key="add_context"
# )


template = """The following is a video concept: {user_prompt}. 
Relevant visual context from uploaded images: {image_context}.
Generate a detailed narrative script incorporating these visual elements."""

add_context = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=['user_prompt', 'image_context'],
        template=template
    ),
    output_key="add_context"
)

def generate_image_sequence(model, base_prompt, num_frames=15):
    images = []
    for i in range(num_frames):
        frame_prompt = f"{base_prompt}"
        generated_images = model.generate_images(
            prompt=frame_prompt,
            number_of_images=3,
            language="en"
        )
        
        if not generated_images:
            st.error(f"Image generation failed for frame {i+1}")
            continue
            
        try:
            pil_image = Image.open(io.BytesIO(generated_images[0]._image_bytes))
            img_array = np.array(pil_image)
            images.append(img_array)
        except IndexError:
            st.error("No images generated - check prompt validity")
            break
            
    return images
def login_page():
    st.sidebar.header("User Login")
    
    user_id = st.sidebar.text_input("Enter your User ID")
    
    if user_id:
        try:
            user_id = int(user_id)
            
            if user_manager.validate_user(user_id):
                st.session_state['user_id'] = user_id
                st.sidebar.success(f"Welcome back, {user_id}!")
                st.rerun()  # Refresh page
            
            else:
                st.sidebar.warning("User ID not found. Would you like to create an account?")
                
                # 🔹 Move preference input outside button check
                preferences = st.sidebar.text_input("Enter your preferences (comma-separated)")
                
                if st.sidebar.button("Create Account") and preferences:
                    if user_manager.create_user(user_id, preferences):
                        st.session_state['user_id'] = user_id
                        st.sidebar.success(f"Account created for {user_id}!")
                        st.rerun()  # Refresh page
                    
        except ValueError:
            st.sidebar.error("Please enter a valid numeric User ID")
# Streamlit UI
st.title("Video Content Generation")

if 'user_id' not in st.session_state:
    login_page()
    
if 'user_id' in st.session_state:  # 로그인 된 경우에만 메인 컨텐츠 표시
    selected_user_id = int(st.session_state['user_id'])


    # Main content
    tab1, tab2, tab3 = st.tabs(["Generate from Keywords", "Custom Prompt", "RAG Image Upload"])

    with tab1:
        
        st.header("Generate Video from Your Keywords")
        top_keywords = user_manager.get_top_keywords(selected_user_id)
        selected_keywords = st.multiselect(
            "Select keywords to generate video (up to 3)", 
            options=top_keywords,
            max_selections=3
        )
        
        if st.button("Generate Video") and selected_keywords:
            with st.spinner("Generating video from selected keywords..."):
                # Retrieve relevant image context
                rag_handler = RAGProcessor()
                image_context = rag_handler.retrieve_context(selected_keywords)
        
                # Generate enhanced prompt
                
                # Combine selected keywords into a single prompt
                combined_prompt = ", ".join(selected_keywords)
                #result = add_context({'user_prompt': combined_prompt})
                result = add_context({
                    'user_prompt': combined_prompt,
                    'image_context': "\n- ".join(image_context)
                })
                video_prompt = result['add_context']
                image_sequence = generate_image_sequence(image_model, video_prompt)
                
                clip = ImageSequenceClip(image_sequence, fps=3)
                clip.write_videofile("output_video.mp4")
                
                st.write("**Selected Keywords:**", ", ".join(selected_keywords))
                st.video("output_video.mp4")
                os.remove("output_video.mp4")
        elif not selected_keywords:
            st.warning("Please select at least one keyword")


    with tab2:
        st.header("Custom Video Generation")
        user_prompt = st.text_input("Describe what you want to create...", 
                               placeholder="A sunset beach, A wave crashing, surfer riding a wave")
    
        if st.button("Generate Custom Video") and user_prompt:
            with st.spinner("Generating custom video..."):
                rag_handler = RAGProcessor()
                image_context = rag_handler.retrieve_context(user_prompt)
                result = add_context({'user_prompt': user_prompt,
                                  'image_context': "\n- ".join(image_context)
                                })
                st.write("**Generated Context:**")
                st.write(result['add_context'].strip())
            
                video_prompt = result['add_context'].strip()
                image_sequence = generate_image_sequence(image_model, video_prompt)
            
                clip = ImageSequenceClip(image_sequence, fps=3)
                clip.write_videofile("output_video.mp4")
            
                st.video("output_video.mp4")
                os.remove("output_video.mp4")

    with tab3:
        st.header("Image-based RAG Context")
        uploaded_images = st.file_uploader("Upload context images (PNG/JPG)", 
                                      type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)
    
        if uploaded_images:
            with st.spinner("Processing RAG images..."):
                # Initialize vector DB connection
                rag_handler = RAGProcessor()
            
                # Store images and generate embeddings
                for img in uploaded_images:
                    image_bytes = img.getvalue()
                    rag_handler.process_image(image_bytes)
            
                st.success(f"Processed {len(uploaded_images)} images for RAG context")  
