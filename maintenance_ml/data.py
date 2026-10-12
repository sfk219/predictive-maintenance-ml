import numpy as np,pandas as pd
FEATURES=["temperature","vibration","runtime","maintenance_age"]
def synthetic_data(n=1000,seed=42):
    if n<100:raise ValueError("at least 100 rows")
    rng=np.random.default_rng(seed)
    t=rng.normal(70,12,n);v=rng.gamma(3,1,n);h=rng.uniform(100,10000,n);a=rng.uniform(1,450,n)
    logit=-5+.05*(t-70)+.5*(v-3)+.0002*(h-5000)+.005*(a-100)
    y=rng.binomial(1,1/(1+np.exp(-logit)))
    return pd.DataFrame({"temperature":t,"vibration":v,"runtime":h,"maintenance_age":a,"failure":y})
