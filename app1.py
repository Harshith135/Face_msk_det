import streamlit as st, numpy as np, tensorflow as tf
from PIL import Image

st.set_page_config(page_title='Face Mask Detection', page_icon='FM', layout='wide')

st.markdown('''
<style>
.main-title {font-size:42px;font-weight:700;margin-bottom:0;}
.sub-text {color:#9aa0a6;margin-bottom:20px;}
.result-box {padding:18px;border-radius:12px;font-size:22px;font-weight:600;color:white;text-align:center;}
.green {background:#15803d;}
.red {background:#b91c1c;}
.metric {background:#1f2937;padding:12px;border-radius:10px;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('face_mask_cnn_model.keras')

model = load_model()

st.markdown("<p class='main-title'>Face Mask Detection System</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload an image to detect whether a person is wearing a face mask.</p>", unsafe_allow_html=True)

file = st.file_uploader('Upload Image', type=['jpg','jpeg','png'])

if file:
    image = Image.open(file).convert('RGB')
    c1, c2 = st.columns([1.2,1])

    with c1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    img = image.resize((128,128))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)
    pred = model.predict(arr)[0][0]

    if pred > 0.5:
        label = 'With Mask'
        conf = pred*100
        color = 'green'
    else:
        label = 'Without Mask'
        conf = (1-pred)*100
        color = 'red'

    with c2:
        st.markdown(f"<div class='result-box {color}'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)
        st.progress(int(conf))

st.markdown('---')
st.caption('Built with TensorFlow and Streamlit')