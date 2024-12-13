#pip install numpy==1.23.5
import numpy as np
from numpy import linalg as LA
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from pymilvus import MilvusClient
import psycopg2
from dotenv import load_dotenv
import os
import time

#Importar el modelo preentrenado de efficientNet
input_shape = (224, 224, 3)
efficientnet_model = EfficientNetB0(weights='imagenet', input_shape=input_shape, include_top=False)
output = GlobalAveragePooling2D()(efficientnet_model.output)
feature_extractor = Model(inputs=efficientnet_model.input, outputs=output)

def extract_features(img_path):
        #response = requests.get(img_path)
        #if response.status_code != 200:
        #    print("Unable to fetch image from URL")
        #    return [] , False
        
        # Load the image into memory
        #img = Image.open(BytesIO(response.content))
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img = img.resize((input_shape[0], input_shape[1]))
        img = img.convert('RGB')
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = feature_extractor.predict(img)
        norm_features = features[0]/LA.norm(features[0])
        return norm_features, True

def main():
    load_dotenv()
    milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))

    # search
    nq = 1
    search_params = {"metric_type": "COSINE"}
    topk = 10  # Request top 10 results
    collection_name = "Caracteristicas_Imagenes"
    
    query_vector, result = extract_features("https://walldeco.com.co/cdn/shop/products/LH_e19a9c92-8e39-4e86-8a0b-0f51f85cde13.png?v=1609712622&width=1260")
    print("Longitud del array features: ", len(query_vector))
    
    t0 = time.time()
    # Search with query_vector wrapped in a list
    results = milvus_client.search(collection_name, [query_vector], limit=topk, search_params=search_params, anns_field="vectorCaracteristicas")
    t1 = time.time()

    print("Top 10 results:")
    for result in results:
        for match in result:  # Each 'result' contains multiple matches
            print(match)  # Access the 'id' of each match
    print(f"search latency: {round(t1-t0, 4)} seconds!")

#Esto evita que main se ejecute al importar
if __name__ == "__main__":
    main()