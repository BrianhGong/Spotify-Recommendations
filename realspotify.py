import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util


def main(client_id, client_secret, your_playlist_name, genre, output_playlist_name):
    sp = api_setup(client_id, client_secret)
    playlist_dic = setup_playlists(sp)
    playlist_df = generate_playlist_df(your_playlist_name, playlist_dic, sp) 
    playlist_df = generate_audio_features_df(playlist_df, sp)
    df = create_comparison_df(genre, sp)
    df = generate_audio_features_df(df, sp)
    recs = create_recommendations(playlist_df, df, sp)
    create_recommendations_playlists(recs, output_playlist_name)

def api_setup(client_id, client_secret):
    
    scope = 'playlist-write-private, playlist-modify-public'
    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret,redirect_uri='http://localhost:8080/')
    sp = spotipy.Spotify(auth=token)
    return sp

def setup_playlists(sp):
    #credit Tanmoy Ghosh for this function and api_setup
    playlist_dic = {}
    for i in sp.current_user_playlists()['items']:
        playlist_dic[i['name']] = i['uri'].split(':')[2]
    return playlist_dic

def generate_playlist_df(playlist_name, playlist_dic, sp):
    playlist = pd.DataFrame()
    tracks = []
    offset = 0
    limit = 100  
    playlist_id = playlist_dic[playlist_name]
    while True:
        response = sp.playlist_tracks(playlist_id, offset=offset, limit=limit)
        items = response['items']
        if not items:
            break
        for item in items:
            tracks.append(item['track'])
        offset += len(items)
    for i, j in enumerate(tracks):
        playlist.loc[i, 'artist'] = j['artists'][0]['name']
        playlist.loc[i, 'track_name'] = j['name']
        playlist.loc[i, 'track_id'] = j['id']
        playlist.loc[i, 'url'] = j['album']['images'][1]['url']
    return playlist

def generate_audio_features_df(playlist_df, sp):
    playlist_df['acousticness'] = None
    playlist_df['danceability'] = None
    playlist_df['energy'] = None
    playlist_df['instrumentalness'] = None
    playlist_df['liveness'] = None
    playlist_df['loudness'] = None
    playlist_df['speechiness'] = None
    playlist_df['tempo'] = None
    playlist_df['valence'] = None
    playlist_df['popularity'] = None
    for index, row in playlist_df.iterrows():
        track_id = row['track_id']
        audio_features = sp.audio_features(track_id)
        track_info = sp.track(track_id)
        if audio_features:
            playlist_df.at[index, 'acousticness'] = audio_features[0]['acousticness']
            playlist_df.at[index, 'danceability'] = audio_features[0]['danceability']
            playlist_df.at[index, 'energy'] = audio_features[0]['energy']
            playlist_df.at[index, 'instrumentalness'] = audio_features[0]['instrumentalness']
            playlist_df.at[index, 'liveness'] = audio_features[0]['liveness']
            playlist_df.at[index, 'loudness'] = audio_features[0]['loudness']
            playlist_df.at[index, 'speechiness'] = audio_features[0]['speechiness']
            playlist_df.at[index, 'tempo'] = audio_features[0]['tempo']
            playlist_df.at[index, 'valence'] = audio_features[0]['valence']
            playlist_df.at[index, 'popularity'] = track_info['popularity']
    scaler = MinMaxScaler()
    columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']
    playlist_df[columns] = scaler.fit_transform(playlist_df[columns])
    playlist_df = playlist_df.drop_duplicates(subset=["loudness"])
    return playlist_df

def generate_playlist_vector(playlist_df):
    feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']
    vector = playlist_df[feature_columns].mean()
    return vector

def create_comparison_df(genre, sp):
    df = pd.DataFrame()
    for i in range(7):
        recommendations = sp.recommendations(seed_genres=[genre], limit=100)
        tracks_data = []
        for track in recommendations['tracks']:
            
            track_data = {
                'track_name': track['name'],
                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                'album': track['album']['name'],
                'track_id': track['id']
            }
            tracks_data.append(track_data)
        iteration_df = pd.DataFrame(tracks_data)
        df = pd.concat([df, iteration_df], ignore_index=True)
    
    return df

def create_recommendations(playlist_df, comparison_playlist, sp):
    import numpy as np
    feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']
    vector = generate_playlist_vector(playlist_df)
    comparison_playlist['cosine_similarity'] = None
    dot_product = np.dot(comparison_playlist[feature_columns].values, vector)
    norm_vector = np.linalg.norm(vector)
    norm_features = np.linalg.norm(comparison_playlist[feature_columns].values, axis=1)
    comparison_playlist['cosine_similarity'] = dot_product / (norm_vector * norm_features)
    comparison_playlist = comparison_playlist.sort_values(by='cosine_similarity', ascending=False)
    comparison_playlist = comparison_playlist[:50]
    return comparison_playlist

def create_recommendations_playlists(recs, playlist_name):
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-modify-private",
            redirect_uri='http://localhost:8080/',
            client_id="10309c00c36c41bc80ffd50ccfcaef5c",
            client_secret="322b870e4d91487f81ce014d9aef65ef",
            cache_path="token.txt"
        )
    )
    results = sp.me()
    results['id']
    sp.user_playlist_create(sp.me()['id'], playlist_name, public=False, collaborative=False, description="Cosine Similarity Playlist!")
    playlist_dic = {}
    for i in sp.current_user_playlists()['items']:
        playlist_dic[i['name']] = i['uri'].split(':')[2]
    playlist_id = playlist_dic[playlist_name]
    sp.user_playlist_add_tracks(user=sp.me()['id'], playlist_id=playlist_id, tracks=recs['track_id'].tolist())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create playlist recommendations based on audio features and genre.")
    parser.add_argument("--client_id", type=str, help="Spotify Client ID")
    parser.add_argument("--client_secret", type=str, help="Spotify Client Secret")
    parser.add_argument("--playlist_name", type=str, help="Your playlist name")
    parser.add_argument("--genre", type=str, help="Genre for recommendations")
    parser.add_argument("--output_playlist_name", type=str, help="Output playlist name")
    args = parser.parse_args()
    main(args.client_id, args.client_secret, args.playlist_name, args.genre, args.output_playlist_name)



