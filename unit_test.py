from helper import prepare_data
from main import SimpleVectorDatabase
import pytest
import os

QUERY = "A cat and a dog are playing together"
sentences = prepare_data()

def test_flat_index():
    
    db = SimpleVectorDatabase(sentences, index_method="flat")

    result = db.search(QUERY, top_k=1)
    assert len(result) == 1
    assert type(result) == list
    assert type(result[0]) == str

def test_ivf_index():
    db = SimpleVectorDatabase(sentences, index_method="ivf")

    result = db.search(QUERY, top_k=1)
    assert len(result) == 1
    assert type(result) == list
    assert type(result[0]) == str

def test_hnsw_index():
    db = SimpleVectorDatabase(sentences, index_method="hnsw")

    result = db.search(QUERY, top_k=1)
    assert len(result) == 1
    assert type(result) == list
    assert type(result[0]) == str

def test_pq_index():
    db = SimpleVectorDatabase(sentences, index_method="pq")

    result = db.search(QUERY, top_k=1)
    assert len(result) == 1
    assert type(result) == list
    assert type(result[0]) == str

def test_sq_index():
    db = SimpleVectorDatabase(sentences, index_method="sq")

    result = db.search(QUERY, top_k=1)
    assert len(result) == 1
    assert type(result) == list
    assert type(result[0]) == str

def test_save_data():
    db = SimpleVectorDatabase(sentences, index_method="flat")
    db.save_database("test_db.pkl", force=True)
    assert os.path.exists("./__cache__/data/test_db.pkl")
    

def test_load_data():
    db = SimpleVectorDatabase(sentences, index_method="flat")
    DATA_DIR = "./__cache__/data"
    db.save_database("test_db.pkl", force=True)
    db.load_database(DATA_DIR + "/test_db.pkl")
    assert db.index_method == "flat"
    assert len(db.input_sentences) == len(sentences)
    assert len(db.sentence_embeddings) == len(sentences)

# def test_diskann_index():
#     db = SimpleVectorDatabase(sentences, index_method="diskann")

#     result = db.search(QUERY, top_k=1)
#     assert len(result) == 1
#     assert type(result) == list
#     assert type(result[0]) == str

if __name__ == "__main__":
    pytest.main(["-v", "-s", "unit_test.py"])




