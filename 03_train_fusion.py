import joblib, lightgbm as lgb
from sklearn.metrics import roc_auc_score

features, labels = joblib.load("cache/features.pkl")
X, y = features, labels
train_data = lgb.Dataset(X, label=y)
params = dict(objective="binary", metric="auc", verbosity=-1)
model = lgb.train(params, train_data, num_boost_round=500)
joblib.dump(model, "models/fusion.pkl")