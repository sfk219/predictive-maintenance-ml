from maintenance_ml.data import synthetic_data
from maintenance_ml.train import train
def test_data():assert len(synthetic_data(100))==100
def test_training():assert 0<=train(250)[1]["roc_auc"]<=1
