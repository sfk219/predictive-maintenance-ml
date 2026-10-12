"""Synthetic predictive-maintenance experiment, not production predictions."""
import json,argparse
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import synthetic_data,FEATURES
def train(n=1000):
    df=synthetic_data(n)
    xtr,xte,ytr,yte=train_test_split(df[FEATURES],df["failure"],test_size=.25,stratify=df["failure"],random_state=42)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
    model.fit(xtr,ytr)
    scores=model.predict_proba(xte)[:,1]
    return model,{"roc_auc":round(float(roc_auc_score(yte,scores)),4),"samples":n,"report":classification_report(yte,model.predict(xte),output_dict=True)}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="artifacts");a=p.parse_args()
    model,metrics=train();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"model.joblib");(out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print("Test AUC:",metrics["roc_auc"])
