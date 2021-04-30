"""Machine learning functions"""

import logging
import pandas as pd
import random
import spotipy

from fastapi import APIRouter
from joblib import load
from os import getenv
from pydantic import BaseModel, Field, validator
from spotipy.oauth2 import SpotifyClientCredentials

log = logging.getLogger(__name__)
router = APIRouter()

# Spotify credentials - this needs to be changed to environment variables
SPOTIFY_API_KEY = 'b480cf11cde74d1b8d645b3f86f38d8b'
SPOTIFY_API_KEY_SECRET = 'd0ed275c85fe48e19bf016c90d83221d'

# load an instance of the spotipy class
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_API_KEY,
                                                           client_secret=SPOTIFY_API_KEY_SECRET))

# load our knn model
classifier = load("knn_model.joblib")

# provided scaffolding class
class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    artist: str = Field(..., example="Foo Fighters")
    track: str = Field(..., example="Everlong")


    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    # new method for making Spotify queries using spotipy
    def to_query(self):
        """Convert pydantic object to Spotify query string"""
        convert2dict = dict(self)
        convert2list = " ".join([f"{k}:{v}" for k, v in convert2dict.items()])
        return convert2list


@router.post('/predict')
async def predict(item: Item):
    """
    Return k-nearest neighbors based on user input - artist & song 🔮

    ### Request Body
    - `artist`: Enter a band or artist name here 
    - `song`: Enter a song recorded by the artist 

    ### Response
    - `prediction`: 10 song / artist parings based on user input

    What a great way to discover new music!
    """
    # import our .csv (database)
    df = pd.read_csv('app/spotify_rock.csv', index_col=0)
    # turn pydantic object into a searchable string
    query = item.to_query()
    # grab search user_input search results from Spotify
    search_results = sp.search(q=query, type="track", limit=1)
    # grab the song ID so we can search it's features
    search_id = search_results['tracks']['items'][0]['id']
    # grab the name of the song (I've been doing this just for verifictation during buildout)
    name = search_results['tracks']['items'][0]['name']
    # grab the release date - it's a model input
    year = int(search_results['tracks']['items'][0]['album']['release_date'][:4])
    # search user_input song features
    search_features = sp.audio_features(tracks=search_id)
    # create dictionary to pass into 'to_df'
    song_cluster = {'duration_ms': search_features[0]['duration_ms'],
                    'explicit': 0,
                    'release_date': year,
                    'danceability': search_features[0]['danceability'],
                    'energy': search_features[0]['energy'],
                    'key': search_features[0]['key'],
                    'loudness': search_features[0]['loudness'],
                    'mode': search_features[0]['mode'],
                    'speechiness': search_features[0]['speechiness'],
                    'acousticness': search_features[0]['acousticness'],
                    'instrumentalness': search_features[0]['instrumentalness'],
                    'liveness': search_features[0]['liveness'],
                    'valence': search_features[0]['valence'], 
                    'tempo': search_features[0]['tempo'],
                    'time_signature': search_features[0]['time_signature']}
    # create single row dataframe to pass into our KNN model
    X_song_cluster = pd.DataFrame([dict(song_cluster)])
    # turn our dataframe into a series object
    series = X_song_cluster.iloc[0, :].to_numpy()
    # query our model for 10 nearest neighbors of user's input song
    neighbors = classifier.kneighbors([series], return_distance=False)
    # index into return object
    answers = df.iloc[neighbors[0], [1, 2]]
    # return answers
    return answers

    # uvicorn app.main:app --reload
