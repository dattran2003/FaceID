import torch
import torch.nn as nn
from config.config import Config
from src.feature_extraction.extract import extract_features
import pandas as pd

def verify(face_features: torch.Tensor, database_features: torch.Tensor, threshoud: float = 0.5):
    if(face_features.numel() == 0):    
        raise ValueError('There is no faces to verify')
    
    if(database_features.numel() == 0):
        raise ValueError('The database is empty')
    
    sim = nn.CosineSimilarity(dim=-1)
    sim_scores = sim(face_features.unsqueeze(1), database_features.unsqueeze(0))

    face_ids = torch.argmax(sim_scores, dim=1)
    face_sims = torch.max(sim_scores, dim=1).values

    filt = face_sims < threshoud
    face_ids[filt] = -1
    return face_ids, face_sims

def load_database_features(path: str):
    pass

def load_identities(path: str):
    return pd.read_csv(path)



def get_identity_info(config: Config):
     
    db_path = config.get(section='path', key='db_path')
    info_path = config.get(section='path', key='identity_info_path')

    # wait to emplement feautre extraction steps
    face_features = extract_features()
    db_features = load_database_features(db_path)
    df_identities = load_identities(info_path)

    face_ids, face_sims = verify(face_features, db_features)
    face_ids = face_ids.tolist()

    verified_identities = df_identities.iloc[face_ids].copy()
    verified_identities['similarity'] = face_sims.numpy()

    return df_identities
