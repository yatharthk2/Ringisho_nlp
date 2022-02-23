import clip
import torch

import numpy as np
import pandas as pd


class ImageApi:
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
        self.photo_ids = list(self.photo_ids['photo_id'])

        # Load the features vectors
        self.photo_features = np.load("unsplash-dataset/features.npy")

        # Convert features to Tensors: Float32 on CPU and Float16 on GPU
        self.photo_features = torch.from_numpy(self.photo_features).float().to(self.device)
        # Print some statistics
        print(f"Photos loaded: {len(self.photo_ids)}")

    def encode_search_query(self, search_query):
        with torch.no_grad():
            # Encode and normalize the search query using CLIP
            text_encoded = self.model.encode_text(
                clip.tokenize(search_query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

        # Retrieve the feature vector
        return text_encoded

    def search_unsplash(self, search_query, results_count):
        # Encode the search query
        text_features = self.encode_search_query(search_query)
        # Find the best matches
        best_photo_ids = find_best_matches(text_features, self.photo_features,
                                           self.photo_ids, results_count)
        # Display the best photos
        return best_photo_ids

    def get_image_url(self, search_query):
        result = self.search_unsplash(search_query, 3)
        return result


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