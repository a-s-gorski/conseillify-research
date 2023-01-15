import reranking
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict
from ...data.make_dataset import load_songs_encodings
from ...features.build_features import save_pickle
import click
import logging
import os
from typing import List, Optional, Dict

def assign_popularity(occurances: int) -> int:
    if occurances <= 2:
        return 0
    elif occurances <= 5:
        return 1
    elif occurances <= 25:
        return 2
    elif occurances <= 100:
        return 3
    else:
        return 4

def encode_features(rec_input: List[str], encoded_features) -> List[str]:
    return list(map(lambda track: encoded_features[track], rec_input))


def reranking_component(rec_input: List[str], features: Dict[str, str], n_recommendations: Optional[int]=100, desired_distrubution: Optional[Dict[str, float]] = {})->List[str]:
    rerank_indices = reranking.rerank(
        features,  # attributes of the ranked items
        desired_distrubution,  # desired item distribution
        max_na=None,  # controls the max number of attribute categories applied
        k_max=None,  # length of output, if None, k_max is the length of `item_attribute`
        algorithm="det_greedy",  # "det_greedy", "det_cons", "det_relaxed", "det_const_sort"
        verbose=True,  # if True, the output is with detailed information
    )
    rerank_indices = rerank_indices[:n_recommendations]
    reranked = list(map(lambda x: rec_input[x], rerank_indices))
    return reranked

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_filepath: str, output_filepath: str):

    logging.info("Loading playlists")
    playlists = pd.read_csv(os.path.join(input_filepath, "playlists.csv")).to_numpy()
    songs_encodings = load_songs_encodings(os.path.join(input_filepath, "songs_encodings.csv"))
    reverse_se = {track_id: track_uri for track_uri, track_id in songs_encodings.items()}
    
    logging.info("Calculating distribution")
    track_popularity = dict(Counter(playlists.flatten()))
    track_popularity.pop(-1, None)
    uri_popularity = {reverse_se[track_id]: track_pop for track_id, track_pop in track_popularity.items()}
    track_popularity_encoded = list(map(assign_popularity, list(track_popularity.values())))
    uris_encoded = np.array([(uri, pop) for uri, pop in zip(uri_popularity.keys(), track_popularity_encoded)])
    uris_dict = {uri: pop for (uri, pop) in uris_encoded}
    desired_distribution = {0:0.2, 1:0.2, 2:0.2, 3:0.2, 4:0.2}

    logging.info("Calculating features")
    test_rec = ['spotify:track:2ZXf1655NlMSZ2L1bMLXsO', 'spotify:track:0cqRj7pUJDkTCEsJkx8snD', 'spotify:track:7wUXSi54XI9HBeeOnLYNTW', 'spotify:track:4a69hgOcOev1vRua1wdKQS', 'spotify:track:2KuSmVNePUicJqBuimpMri', 'spotify:track:0CSZ1DMvRZH0WLB5V0WSom', 'spotify:track:3vMrcGW4o35zY6vXjWb1p7', 'spotify:track:7LNzXINzIYfmqF1Zy3uceA', 'spotify:track:1IhqWNtaLrr2QdhNn6mxu2', 'spotify:track:1hdVwjwMwW5uLOyCMD4B4Z', 'spotify:track:1k7CqKKwQu4zIK6DrsuqKG', 'spotify:track:1cCYNgQILq8D3uX2sy50Mv', 'spotify:track:3quUoHAWK71g97jH5aeYIH', 'spotify:track:2K1meYaIxerapOrEief7JH', 'spotify:track:0vmaQUwLLdJb9Fy61OkXw5', 'spotify:track:7BKxjTSDtobH427jnfky6m', 'spotify:track:3qzVJh6INW1CzSDVR9MRgS', 'spotify:track:2d9dbonEAbSgYAPLeFRbPc', 'spotify:track:0aHWzIvtnkG3FVcBoFQ6BG', 'spotify:track:5gdeC9k5eNW0vey8v0t3bg', 'spotify:track:5jHYEOabEw2RSwOzf8nQPn', 'spotify:track:0cJVSSihsr6iNDOGfeumJV', 'spotify:track:074XKhqjJ6SuYb2YsdMNQK', 'spotify:track:1PpfeNl5uYRBhfFcig2Uz6', 'spotify:track:1Z8bkmu5NfFSQRl3GOoOIl', 'spotify:track:6wWphSJNz9GiKWb6EcealX', 'spotify:track:2itW6yNVqzqtFM3G3qbhpx', 'spotify:track:6kdjd3IBSIjMWZEm5BFzNH', 'spotify:track:3DBoCDi9KqMVSP2rm8LE0Q', 'spotify:track:5m6y5b6M8PuNz1wy6FLWgo', 'spotify:track:5SUeM4FimNghVHQPsYhPO9', 'spotify:track:2SImdxjfObHtQYN9zgJJXh', 'spotify:track:3JZDax8w9XaM2SXVim3lxd', 'spotify:track:66yzHUHNo6Rg1wbpFjJ1MH', 'spotify:track:5PhpcLNSV9DQPwBNag8dRH', 'spotify:track:0tY264U9InKFkHKz4yj52H', 'spotify:track:3oDFtOhcN08qeDPAK6MEQG', 'spotify:track:3e0WR73qub5sz3ggtbQuPu', 'spotify:track:7fOGb4nnR5saAEq5twEOqE', 'spotify:track:6LtpK0EOUJsbTD6bKlaXkZ', 'spotify:track:2xYlyywNgefLCRDG8hlxZq', 'spotify:track:1uDjaezEbalGyGnuH80zDK', 'spotify:track:1prNlxt0CJBXEC8Xp0wxYd', 'spotify:track:2W4GuOrUp7KEBcchAH8pmu', 'spotify:track:6V5nMvQwsiyW7rEyZq3gfQ', 'spotify:track:0mcWFESxnymoKwFzh6PmUT', 'spotify:track:2oaUcgHUEsK4QJ6onplT5W', 'spotify:track:500XjFuAZEBODSL6boVKbx', 'spotify:track:57NzzfHfE8sOSbIHRJJYv7', 'spotify:track:5kBXJ9V4i822ZdyHgwSh0o', 'spotify:track:1weEqgqTn03x8hmJ2mHMET', 'spotify:track:6leyjnWsVtoduXRTVIyCn1', 'spotify:track:2nbyqNoFCgpFHBK8syBOD0', 'spotify:track:1Z7ecWxelX9nw0xQJuYLLZ', 'spotify:track:2NELtMgQ8HSdrGrYQPLnC3', 'spotify:track:2hvXynoLGdktCusZwX7uzN', 'spotify:track:6MP4P1wp8gL8zey9YRPh7T', 'spotify:track:0a9N94pHEOmE2xLBsygyy7', 'spotify:track:3ijA4bRNDhnxZhkdbDYMjY', 'spotify:track:5Y6iYiErdkFwye1YD7GKHj', 'spotify:track:1xIvb2dl9aHzGjdsnWqOhG', 'spotify:track:7f22dVIj5ZdRO96iWySrOv', 'spotify:track:2oHmRO0X4gW4VlCJRcZzez', 'spotify:track:0vzSdr6SsnskLoNgtpqtVU', 'spotify:track:1NeKY4qWsCnhY8fvhwQa7q', 'spotify:track:4hRSKLxWptKUvKF7wr60B2', 'spotify:track:0eUJzAGs9OKSOmLVpsng7e', 'spotify:track:1mvEbRAlocvkJvqZIj3zHu', 'spotify:track:32a5eLrlrKiXrqb1D16CUY', 'spotify:track:4CndyuajS0lmZDLzPSt039', 'spotify:track:6OVwDWn2LycmfrJJ4lvDqF', 'spotify:track:2JtzwKuk3RmxDSeoY6LMMI', 'spotify:track:5Sl1HOqv2saJuPRnhxLO06', 'spotify:track:148wRodgklV4oYyXcX1Bkg', 'spotify:track:1ZEVSOFaWo6hWpuSsx1Xwr', 'spotify:track:3mkNZA4Q0dYvZIn6OZEmOz', 'spotify:track:2sywvGUBuSuYICHHKOElSn', 'spotify:track:6rS9yOTShKlU8izy5ZUjQC', 'spotify:track:0zF5tQfC9wuZxmN1TJ5hNk', 'spotify:track:5g3paAWMZ7XJIlFwI8qrz4', 'spotify:track:4lrM4VDylnOenu5bxDwHzq', 'spotify:track:53wHJtuT9j21joF1OrXk42', 'spotify:track:3ZJAp1N2Stlo2ixvn9rmuM', 'spotify:track:61iyAeMVUNu7vIVqedsBFB', 'spotify:track:0g9IOJwdElaCZEvcqGRP4b', 'spotify:track:0hB7p4rUwkpVyNifHxTXXT', 'spotify:track:1wXuogT7bvqnhuWzDBNOdV', 'spotify:track:5KnweDmP8kD4DgvWCt3ICh', 'spotify:track:025IP3BitccPpq8Ybgxnlh', 'spotify:track:2GjMLXTshHSCIlWYqqWdBO', 'spotify:track:2b0dKwpLJlM5wjNgS03lLx', 'spotify:track:4789JLZ8VZBD3NoEcpOrSi', 'spotify:track:77QhwObfiZVV8Vsa9qCUqq', 'spotify:track:70WmnHlwMYW8Ss7AshcgAE', 'spotify:track:4GLQs3YHKSFXBuvdvlGpmO', 'spotify:track:19BDOJWYo22GHH5ASvzkew', 'spotify:track:3gr540RhkP9dXsGEBpwawZ', 'spotify:track:1BOH9Tv52qRLNJNaAVkVmZ', 'spotify:track:1hWCBIhwfUFwhENSjY1BUq', 'spotify:track:30aaR9wCjfPqQSDqbhsTaw']
    features = encode_features(test_rec, uris_dict)

    logging.info("Reranking")
    reranked = reranking_component(test_rec, features, 20, desired_distribution)

    logging.info(f"Reranked recommendations: {reranked}")

    logging.info("Saving artifacts")
    save_pickle(uris_dict, os.path.join(output_filepath, "uris_dict.pkl"))
    save_pickle(desired_distribution, os.path.join(output_filepath, "desired_distribution.pkl"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
