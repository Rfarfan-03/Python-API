from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ImageCompare import extract_features
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
app = FastAPI()


@app.post("/getSimilarProducts")
async def Compare(request: Request):
    load_dotenv()
    milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    #The body contains the bytes for the image
    body = await request.body()
    # search
    nq = 1
    search_params = {"metric_type": "COSINE"}
    topk = 10  # Request top 10 results
    collection_name = "Caracteristicas_Imagenes"
    
    query_vector, result = extract_features(body)
    print("Longitud del array features: ", len(query_vector))
    
    
    # Search with query_vector wrapped in a list
    results = milvus_client.search(collection_name, [query_vector], limit=topk, search_params=search_params, anns_field="vectorCaracteristicas")
    
    matching = {"Resultados" : []}


    
    print("Top 10 results:")
    for result in results:
        for match in result:  # Each 'result' contains multiple matches
            print(match)  # Access the 'id' of each match
            product = {
                "identifier" : match["id"]
            }
            matching["Resultados"].append(product)
    
    print(matching)
            
    
    return matching 

    