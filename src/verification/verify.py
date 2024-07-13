import torch
import torch.nn as nn

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

