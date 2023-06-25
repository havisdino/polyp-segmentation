import streamlit as st
from utils import *


st.set_page_config(
    page_title='Polyp segmentation',
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('Polyp segmentation')

img = st.file_uploader('Upload your endoscopic image')

img = decode_and_normalize_image(img.read())
