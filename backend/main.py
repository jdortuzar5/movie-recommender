from fastapi import FastAPI, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import pandas as pd
import json
import uvicorn
import uuid
from bson import json_util
from typing import List
import requests
import numpy as np
import os
import time

MONGO_HOST = os.getenv("MONGOHOST")
# TF_SERVING = "localhost"
TF_SERVING = os.getenv("TF_SERVING_HOST")

mongo_client = MongoClient(MONGO_HOST, 27017)
db = mongo_client["MovieRecommenderDB"]
movie_col = db["movies"]
user_col = db["user"]
status_col = db["status"]
ratings_col = db["ratings"]
movie_encoded_col = db["movieEncoding"]

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_number_of(collection):
    num_of_movies = collection.find({}).count()
    return num_of_movies

def save_to_db(csv_file, file_size, jobId, collection):
    _index = get_number_of(collection)
    csv_file.file.seek(0)
    for i, l in enumerate(csv_file.file):
        pass
    csv_file.file.seek(0)
    headers = csv_file.file.readline().decode().replace(";\n", "").split(";")
    for j, l in enumerate(csv_file.file):
        line = l.decode()
        line = line.replace(";\n", "")
        row_elem = line.split(";")
        if len(row_elem) > len(headers):
            job_doc = {"jobId": jobId,
                "status": "Error",
                "percentage": int((j/i)*100),
                "reason": f"Ilegal Character in line {j}"}
            status_col.update_one({"jobId": jobId}, {"$set": job_doc})
        else:
            doc = {}
            for e in range(len(row_elem)):
                doc[headers[e]] = row_elem[e]
            doc["index"] = _index + j
            if collection.find_one(doc) is None: 
                collection.insert_one(doc)
            else:
                pass
            status_col.update_one({"jobId": jobId}, {"$set": {"percentage": int((j/i)*100)}})
    status_col.update_one({"jobId": jobId}, {"$set": {"percentage": 100, "status": "complete", "fileName": csv_file.filename, "fileSize": file_size, "numOfRows": i}})
            
def make_movie_encoding():
    unique_movies = ratings_col.distinct("movieId")
    movie_encoder = {x: i for i, x in enumerate(unique_movies)}
    for key, item in movie_encoder.items():
        doc = {"movieId": key, "index": item}
        movie_encoded_col.insert_one(doc)

def save_ratings_to_db(csv_file, file_size, jobId):
    save_to_db(csv_file, file_size, jobId, ratings_col)
    make_movie_encoding()


@app.get("/status/{jobId}")
def get_status_bulk_update(jobId: str):
    job_doc = status_col.find_one({"jobId": jobId}, {'_id': False})
    job_doc = json.loads(json_util.dumps(job_doc))
    return job_doc

def insert_status(doc):
    status_col.insert_one(json.loads(json.dumps(doc)))

@app.post("/movies/bulk_update")
def bulk_update_movie_database(background_task: BackgroundTasks, csv_file: UploadFile = File(...)):
    jobId = str(uuid.uuid4())
    csv_file.file.seek(0, 2)
    file_size = csv_file.file.tell()/1000
    job_doc = {"jobId": jobId,
                "status": "inProgress",
                "percentage": 0}
    insert_status(job_doc)
    background_task.add_task(save_to_db, csv_file, file_size, jobId, movie_col)

    return {"filename": csv_file.filename,
            "file_size": file_size,
            "job": job_doc}

@app.post("/ratings/bulk_update")
def bulk_update_rating_database(background_task: BackgroundTasks, csv_file: UploadFile = File(...)):
    jobId = str(uuid.uuid4())
    csv_file.file.seek(0, 2)
    file_size = csv_file.file.tell()/1000
    job_doc = {"jobId": jobId,
                "status": "inProgress",
                "percentage": 0}
    insert_status(job_doc)
    background_task.add_task(save_ratings_to_db, csv_file, file_size, jobId)

    return {"filename": csv_file.filename,
            "file_size": file_size,
            "job": job_doc}

def find_movies_by_ids(id_list):
    title_list = []
    for id in id_list:
        movie_title = movie_col.find_one({"movieId": str(id)})
        if movie_title is not None:
            title_list.append(movie_title["title"])
        else:
            pass
    return title_list

def get_recomendation(movie_ids):
    movie_array = np.hstack(([[0]]*len(movie_ids), movie_ids))
    body = {"instances": movie_array.tolist()}
    url = f"http://{TF_SERVING}:8501/v1/models/movie_model:predict"
    response = requests.request("POST", url, data=json.dumps(body))
    aux = response.json()
    return aux

def get_movie_index(movieIds):
    movie_indexs = []
    for movie in movieIds:
        movie_doc = movie_col.find_one({"movieId": str(movie)})
        if movie_doc is not None:
            movie_indexs.append([movie_doc["index"]])
        else:
            pass

    return movie_indexs

def encode_movieIds(movieIds):
    encoded_movies = []

    for movie in movieIds:
        doc = movie_encoded_col.find_one({"movieId": str(movie)})
        if doc is not None:
            encoded_movies.append([doc["index"]])
        else:
            pass
        
    return encoded_movies

def find_movies_not_watched(movieIndexs):
    indexs_to_watch = []
    movies_to_watch = movie_col.find({"$nor": [{"movieId": {"$in": movieIndexs}}]})
    indexs_to_watch = [int(x["movieId"]) for x in movies_to_watch]
    return indexs_to_watch

def clean_up_recommendations(recommendation_scores, top_indexes):
    recommendation_body = []
    for index in top_indexes:
        movieId = movie_encoded_col.find_one({"index": int(index)}, {'_id': False})["movieId"]
        movieDoc = movie_col.find_one({"movieId": movieId}, {'_id': False})
        movieScore = recommendation_scores[index]
        body = {
            "title": movieDoc["title"],
            "genre": movieDoc["genres"],
            "movieScore": movieScore[0]
        }
        recommendation_body.append(body)
    return recommendation_body


def generate_recommendations(movieIds, jobId):
    start = time.time()
    movieIds = [str(x) for x in movieIds]
    still_to_watch_indxs = find_movies_not_watched(movieIds)
    encoded_movies = encode_movieIds(still_to_watch_indxs)
    recommendation = get_recomendation(encoded_movies)
    recomender_scores = np.array(recommendation["predictions"]).flatten()
    top_recommendations_indx  = np.array(recomender_scores).argsort()[-10:][::-1]
    recommendations = clean_up_recommendations(recommendation["predictions"], top_recommendations_indx)
    inputMovieTitles = find_movies_by_ids(movieIds)
    end = time.time()
    timeTaken = end-start
    status_col.update_one({"jobId": jobId}, {"$set": {"status": "complete", "input": inputMovieTitles, "recommendation": recommendations, "timeTaken": timeTaken}})


@app.post("/movie/make_recom")
def make_recomendation(movies: List, background_task: BackgroundTasks):
    jobId = str(uuid.uuid4())
    job_doc = {"jobId": jobId,
                "status": "inProgress"
                }
    insert_status(job_doc)
    background_task.add_task(generate_recommendations, movies, jobId)
    
    return job_doc

@app.get("/autocomplete")
def get_autocomplete_movies():
    movie_all = movie_col.find()
    data = {}
    for doc in movie_all:
        data[doc["title"]] = doc["movieId"]
    return data

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")