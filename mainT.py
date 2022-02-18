import clip
import torch
import requests
import urllib.request

import numpy as np
import pandas as pd

from pathlib import Path
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from IPython.display import Image
from IPython.display import display
from IPython.core.display import HTML
from fastapi.responses import FileResponse 


def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(
            clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    # Retrieve the feature vector
    return text_encoded

def find_best_matches(text_features,
                      photo_features,
                      photo_ids,
                      results_count):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    similarities = (photo_features @ text_features.T).squeeze(1)

    # Sort the photos by their similarity score
    best_photo_idx = (-similarities).argsort()

    # Return the photo IDs of the best matches
    return [photo_ids[i] for i in best_photo_idx[:results_count]]

def display_photo(photo_id):
    # Get the URL of the photo resized to have a width of 320px
    photo_image_url = f"https://unsplash.com/photos/{photo_id}/download?w=320"

    # Display the photo
    display(Image(url=photo_image_url))

    # Display the attribution text
    display(HTML(f'Photo on <a target="_blank" href="https://unsplash.com/photos/{photo_id}">Unsplash</a> '))
    print()

def search_unsplash(search_query, photo_features, photo_ids, results_count):
    
    # Encode the search query
    text_features = encode_search_query(search_query)

    # Find the best matches
    best_photo_ids = find_best_matches(text_features, photo_features,
                                       photo_ids, results_count)

    # Display the best photos
    return best_photo_ids


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def doStuff(search_query):

    photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
    photo_ids = list(photo_ids['photo_id'])

    # Load the features vectors
    photo_features = np.load("unsplash-dataset/features.npy")

    # Convert features to Tensors: Float32 on CPU and Float16 on GPU
    if device == "cpu":
        photo_features = torch.from_numpy(photo_features).float().to(device)
    else:
        photo_features = torch.from_numpy(photo_features).to(device)

    # Print some statistics
    print(f"Photos loaded: {len(photo_ids)}")

    #search_query = "should i buy the iphone 13 or pixel 6"

    result = search_unsplash(search_query, photo_features, photo_ids, 1)

    return result


# app = FastAPI()

# class Info(BaseModel):
#     user_id : int
#     question_id : int
#     question: str

# @app.post("/")
# def read_root(info : Info):

#     best_photo_id = doStuff(info.question)

#     photo_image_url_1 = "https://unsplash.com/photos/"
#     photo_image_url_2 = "/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjQ0ODE4NTk4&force=true"

#     return {
#         "status" : "SUCCESS",
#         "url" : photo_image_url_1 + best_photo_id[0] + photo_image_url_2,
#         "data" : info
#     }