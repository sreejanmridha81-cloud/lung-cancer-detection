import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
model=load_model("model12")
class_indices=pickle.load(open("class_indices12",'rb'))
def load_and_preprocess_image(img_path,target_size=(224,224)):
 img=Image.open(img_path).convert("RGB")
 img=img.resize(target_size)
 img_array=np.array(img)
 img_array=np.expand_dims(img_array,axis=0)
 img_array=img_array.astype("float32")/255.0
 return img_array


def predict_img_class(model,img_path,class_indices):
  preprocessed_img=load_and_preprocess_image(img_path)
  prediction=model.predict(preprocessed_img)
  prediction_class_index=np.argmax(prediction,axis=1)[0]
  predict_lung_cancer=class_indices[int(prediction_class_index)]
  return predict_lung_cancer

st.title("Cancer Detection Model🫁")
file_uploader=st.file_uploader("upload a file....",type=['jpg','jpeg','png'])

if file_uploader is not None:
  st.image(file_uploader,caption='upload a image')
  result=predict_img_class(model,file_uploader,class_indices)
  st.success(f"the lung  is : {result}")
  
