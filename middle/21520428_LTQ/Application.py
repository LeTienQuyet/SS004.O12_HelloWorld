import tensorflow as tf
import numpy as np
import time
from PIL import Image
import random
import streamlit as st

author = "Lê Tiến Quyết"
id = "21520428"

path_dog = ['DOG_ANS.png','DOG_ANS_1.jpg']
path_cat = ['CAT_ANS.jpg','CAT_ANS_1.png','CAT_ANS_2.png','CAT_ANS_3.png']
path_dog_ans = random.choice(path_dog)
path_cat_ans = random.choice(path_cat)

def input_process(path):
    img = Image.open(path)
    img_origi = np.array(img)
    img_resized = np.array(Image.fromarray(img_origi).resize((200,200)))
    img_resized = np.array(img_resized)
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

model = tf.keras.models.load_model('MyModel.keras')

st.title("Kỹ năng nghề nghiệp - SS004.O12")
st.caption(f'Người thực hiện: {author} - {id}')

file = st.file_uploader('Tải ảnh lên', type=['jpg', 'jpeg', 'png', 'jfif'])

if file is not None:
    path_img = file.name
    img = Image.open(path_img)

    input = input_process(path_img)
    output = model.predict(input)

    column_1, column_2 = st.columns(2)
    with column_1:
        st.header('Mô hình dự đoán')
        
        with st.spinner('Loading'):
            time.sleep(3)

        if output >= 0.5:
            dog_ans = Image.open(path_dog_ans)
            st.image(dog_ans, 'Con Chó')
        else:
            cat_ans = Image.open(path_cat_ans)
            st.image(cat_ans, 'Con Mèo')

    with column_2:
        st.header('Kết quả thực tế')
        st.image(img)
