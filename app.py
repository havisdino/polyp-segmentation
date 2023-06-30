import streamlit as st
from utils import *
from model import UNet


st.set_page_config(
    page_title='Polyp segmentation',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)


@st.cache_resource
def load_unet():
    net =  UNet()
    state_dict = torch.load('bin/unet128.pt', 'cpu')
    net.load_state_dict(state_dict)
    net.eval()
    print('model loaded')
    return net


IMG_SIZE = (128, 128)
net = load_unet()


st.title('Polyp segmentation')
c1, c2 = st.columns([0.3, 0.7])

img = c1.file_uploader('Upload your endoscopic image', type=['png', 'jpg', 'jpeg'])
clicked = c1.button('Continue')

if img is not None and clicked:
    try:
        img_org = decode_and_normalize_image(img.read())
    except RuntimeError:
        st.warning('Unsupported image file. Only jpg, jpeg and png are supported.')
    
    img = transforms.functional.resize(img_org, IMG_SIZE, antialias=True)  
    with torch.no_grad():
        mask = net(img)
    
    mask_org = mask.unsqueeze(0)
    mask_org = transforms.functional.resize(mask_org, img_org.shape[2:], antialias=True)
    
    c21, c22 = c2.columns(2)
    
    img, seg, prob, bm = merge(img_org, mask_org)
    
    c21.text('Input image')
    c21.image(img)
    c21.text('Probabilistic mask')
    c21.image(prob)
    
    c22.text('Heat map')
    c22.image(seg)
    c22.text('Binary mask')
    c22.image(bm)
