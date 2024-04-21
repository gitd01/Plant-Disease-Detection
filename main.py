import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr]) # Convert single image to a Batch
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction)
    return result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Detection"])

#Home Page
if(app_mode=="Home"):
    st.header("Plant Disease Detection System")
    image_path="home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    # Welcome to the Plant Disease Detection System! 🌿🔍

Welcome to the Plant Disease Detection System, where our mission is to help identify plant diseases efficiently. By uploading an image of a plant, our system can analyze it to detect any signs of diseases. Let's work together to protect our crops and ensure a healthier harvest!

## How It Works

1. **Upload Image:** Navigate to the **Disease Detection** page and upload an image of a plant suspected to have diseases.
2. **Analysis:** Our system processes the uploaded image using advanced algorithms to identify potential diseases.
3. **Results:** Instantly view the results.

## Why Choose Us?

- **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
- **User-Friendly:** Our interface is designed to be simple and intuitive, providing a seamless user experience.
- **Fast and Efficient:** Receive results within seconds, enabling quick decision-making.

## Get Started

To begin, click on the **Disease Detection** page in the sidebar to upload an image and experience the power of our Plant Disease Detection System!

## About Dataset
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found here 👉 [Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download).
                
This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
A new directory containing 33 test images is created later for prediction purpose.
#### Content
    1. train (70295 images)
    2. test (33 images)
    3. validation (17572 images)

GitHub Repository: [Plant Disease Detection](https://github.com/gitd01/Plant-Disease-Detection)

""")

#About Page
if(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Us

We are a group of final year B.Tech students from Harcourt Butler Technical University working on a project titled "Plant Disease Detection using Deep Learning". The Plant Disease Detector was developed under the guidance of **Dr. Vandana Dixit Kaushik** ma'am. Our team is passionate about leveraging machine learning techniques to address real-world challenges in agriculture, particularly in the detection and prevention of plant diseases.

Our project is being developed as part of our college curriculum. We aim to contribute to the advancement of agricultural technology by providing farmers with an efficient tool for identifying and managing crop diseases. Through our project, we strive to make a positive impact on crop yields, agricultural sustainability, and food security.

#### Team Members
- Divyanshi Gupta
- Himanshu Singh
- Himanshi Rathore
- Vanshika Garg


#### Project Mentor
Dr. Vandana Dixit Kaushik  
Professor, Department of Computer Science And Engineering  
HBTU, Kanpur

---

Feel free to reach out to us if you have any questions or would like to learn more about our project.

**For Queries:**  
You can email us at: [guptadivyanshi047@gmail.com](mailto:guptadivyanshi047@gmail.com)

                """)

#Prediction Page  
elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        #st.snow()
        with st.spinner("Please wait.."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))