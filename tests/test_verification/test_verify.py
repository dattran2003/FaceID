import pytest 
import torch
from FaceID.src.verification.verify import verify 

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_verify_2faces_in_database():
    face_features = torch.tensor([[1, -0.5, 1],
                                  [0.9, 2.1, 3]], dtype=torch.float32)
    database_features = torch.tensor([[1, 2, 3],
                                      [1, -1, 1],
                                      [-1, -2, 3]], dtype=torch.float32)

    face_ids, face_sims = verify(face_features, database_features)
    expected_ids = torch.tensor([1, 0]) 
    expected_sims = torch.tensor([0.9622, 0.9993])

    dif_sims = expected_sims - face_sims
    dif_sims.abs_()

    epsilon = torch.tensor(0.0001) 
    assert torch.equal(face_ids, expected_ids) and torch.all(dif_sims < epsilon)

def test_verify_2faces_include_1face_not_in_database():
    face_features = torch.tensor([[1, -0.5, 1],
                                  [-1, -1, -1]], dtype=torch.float32)
    database_features = torch.tensor([[1, 2, 3],
                                      [1, -1, 1],
                                      [-1, -2, 3]], dtype=torch.float32)

    face_ids, face_sims = verify(face_features, database_features)
    expected_ids = torch.tensor([1, -1]) 
    expected_sims = torch.tensor([0.9622, 0.0])

    dif_sims = expected_sims - face_sims
    dif_sims.abs_()

    epsilon = torch.tensor(0.0001) 
    assert torch.equal(face_ids, expected_ids) and torch.all(torch.where(dif_sims < epsilon, True, False))

def test_verify_faces_when_database_empty():
    face_features = torch.tensor([[1, -0.5, 1],
                                  [-1, -1, -1]], dtype=torch.float32)
    database_features = torch.empty((0, 3), dtype=torch.float32)

    with pytest.raises(ValueError):
        face_ids, face_sims = verify(face_features, database_features)
    

def test_verify_faces_when_there_no_faces():
    face_features = torch.empty((0, 3), dtype=torch.float32)
    database_features = torch.tensor([[1, 2, 3],
                                      [1, -1, 1],
                                      [-1, -2, 3]], dtype=torch.float32)

    with pytest.raises(ValueError):
        face_ids, face_sims = verify(face_features, database_features)
