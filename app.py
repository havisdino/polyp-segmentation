import streamlit as st
from utils import *


print('reloaded')
st.set_page_config(
    page_title='Polyp segmentation',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('Polyp segmentation')

img = st.file_uploader('Upload your endoscopic image')
clicked = st.button('Continue')

if img is not None and clicked:
    try:
        img = decode_and_normalize_image(img.read())
    except RuntimeError:
        st.warning('Unsupported image file. Only jpeg and png are supported.')
    
