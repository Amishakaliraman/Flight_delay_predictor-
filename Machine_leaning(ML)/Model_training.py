# ml/model_training.py
# Trains 3 ML models, compares them, saves the best one

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import time

os.makedirs('streamlit_app', exist_ok=True)

# Load the clean data from Phase 3
print("Loading data...")
df = pd.read_csv('data/flights_clean.csv')
print(f"Loaded: {len(df):,} rows")

# FEATURES = columns the model uses to make predictions
# TARGET   = the column it is predicting (is_delayed)
# Rule: only use encoded columns (_enc) not text columns
# Rule: don't include arr_delay — that's cheating (you can't
#       know arrival delay before the flight lands)
df['is_peak_hour']= df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

FEATURES = ['hour', 'day_of_week', 'month',
            'distance', 'airline_enc', 'origin_enc', 'dest_enc',
            'is_weekend', 'is_peak_hour', 'is_long_haul',
            'season_enc', 'tod_enc']

X = df[FEATURES]
y = df['is_delayed'].astype(int)

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Target balance: {y.mean()*100:.1f}% delayed, {(1-y.mean())*100:.1f}% on time")


# ── Train / Test Split ───────────────────────────────────
# test_size=0.2    = use 20% for testing, 80% for training
# stratify=y       = make sure both splits have same % of
#                    delayed flights (important for balance)
# random_state=42  = makes split reproducible every time
#                    (same rows go to same split every run)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTrain/Test Split:")
print(f"  Training rows : {len(X_train):,} (80%)")
print(f"  Testing rows  : {len(X_test):,}  (20%)")
print(f"  Train delay % : {y_train.mean()*100:.1f}%")
print(f"  Test delay %  : {y_test.mean()*100:.1f}%  (should match train)")

# Dictionary to store results of all 3 models
results = {}

# ── MODEL 1: Logistic Regression ─────────────────────────
# StandardScaler needed for Logistic Regression
# It converts all numbers to same scale so no one feature
# dominates just because it has bigger numbers

print("\n[1/3] Training Logistic Regression...")
t0 = time.time()

scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_train)  # fit on train, transform train
X_te_sc   = scaler.transform(X_test)       # only transform test (no fit!)

lr = LogisticRegression(max_iter=200, random_state=42, n_jobs=-1)
lr.fit(X_tr_sc, y_train)

lr_preds = lr.predict(X_te_sc)
lr_proba = lr.predict_proba(X_te_sc)[:, 1]

acc = accuracy_score(y_test, lr_preds) * 100
auc = roc_auc_score(y_test, lr_proba)  * 100
results['Logistic Regression'] = {
    'accuracy': acc, 'roc_auc': auc,
    'time': round(time.time()-t0, 1)
}
print(f"  Accuracy : {acc:.1f}%")
print(f"  ROC-AUC  : {auc:.1f}%")
print(f"  Time     : {results['Logistic Regression']['time']}s")

# ── MODEL 2: Random Forest ────────────────────────────────
# n_estimators=100  = build 100 trees (more = better but slower)
# max_depth=12      = each tree can be at most 12 levels deep
#                     (prevents overfitting / memorising)
# min_samples_leaf=50 = each leaf must have at least 50 flights
#                       (prevents model from being too specific)
# n_jobs=-1         = use ALL CPU cores to train faster

print("\n[2/3] Training Random Forest (may take 2-4 mins)...")
t0 = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # no scaler needed for Random Forest

rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, rf_preds) * 100
auc = roc_auc_score(y_test, rf_proba)  * 100
results['Random Forest'] = {
    'accuracy': acc, 'roc_auc': auc,
    'time': round(time.time()-t0, 1)
}
print(f"  Accuracy : {acc:.1f}%")
print(f"  ROC-AUC  : {auc:.1f}%")
print(f"  Time     : {results['Random Forest']['time']}s")

# ── MODEL 3: XGBoost ──────────────────────────────────────
# n_estimators=200    = build 200 trees (more than RF)
# max_depth=6         = shallower trees (XGBoost works better shallow)
# learning_rate=0.1   = how much each new tree corrects previous ones
#                       (lower = more careful = better but slower)
# subsample=0.8       = each tree sees 80% of rows randomly
# colsample_bytree=0.8= each tree sees 80% of columns randomly
# eval_metric='logloss' = how it measures error during training

print("\n[3/3] Training XGBoost (may take 2-3 mins)...")
t0 = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0   # silence XGBoost's own print messages
)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, xgb_preds) * 100
auc = roc_auc_score(y_test, xgb_proba)  * 100
results['XGBoost'] = {
    'accuracy': acc, 'roc_auc': auc,
    'time': round(time.time()-t0, 1)
}
print(f"  Accuracy : {acc:.1f}%")
print(f"  ROC-AUC  : {auc:.1f}%")
print(f"  Time     : {results['XGBoost']['time']}s")

# ── Compare all 3 models ──────────────────────────────────
print("\n" + "="*55)
print("MODEL COMPARISON")
print("="*55)
print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10} {'Time':>8}")
print("-"*55)

best_name = None
best_auc  = 0
for name, r in results.items():
    is_best = r['roc_auc'] == max(v['roc_auc'] for v in results.values())
    marker  = " BEST" if is_best else ""
    print(f"{name:<22} {r['accuracy']:>9.1f}% {r['roc_auc']:>9.1f}% {r['time']:>7.1f}s{marker}")
    if r['roc_auc'] > best_auc:
        best_auc  = r['roc_auc']
        best_name = name

# Pick the best model object
best_model = {
    'Logistic Regression': lr,
    'Random Forest':       rf,
    'XGBoost':             xgb_model
}[best_name]

# Save everything the Streamlit app needs in one bundle:
# - the trained model itself
# - the scaler (needed if Logistic Regression won)
# - the feature list (so app knows which columns to use)
# - accuracy and roc_auc (for display in the app)
joblib.dump({
    'model':      best_model,
    'scaler':     scaler,
    'features':   FEATURES,
    'model_name': best_name,
    'accuracy':   results[best_name]['accuracy'],
    'roc_auc':    results[best_name]['roc_auc']
}, 'streamlit_app/model.pkl')

print(f"\nBest model: {best_name}")
print(f"Saved to  : streamlit_app/model.pkl")
print("model_training.py complete!")


# ml/model_evaluation.py
# Deep evaluation of the saved best model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)
import os
os.makedirs('Outputs/charts', exist_ok=True)

# Load the saved model bundle
print("Loading saved model...")
bundle   = joblib.load('streamlit_app/model.pkl')
model    = bundle['model']
FEATURES = bundle['features']
print(f"Model: {bundle['model_name']}")
print(f"Accuracy: {bundle['accuracy']:.1f}%  ROC-AUC: {bundle['roc_auc']:.1f}%")

# Recreate the same test set (same random_state=42 = same rows)
df = pd.read_csv('data/flights_clean.csv')
df['is_peak_hour']= df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
X  = df[FEATURES]
y  = df['is_delayed'].astype(int)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                       stratify=y, random_state=42)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# Print full classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, preds,
      target_names=['On Time', 'Delayed']))

# ── Chart: Confusion Matrix + ROC Curve side by side ────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=['On Time','Delayed']).plot(
    ax=axes[0], cmap='Blues')
axes[0].set_title(f'Confusion Matrix — {bundle["model_name"]}',
                  fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc     = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='#F44336', lw=2,
             label=f'ROC AUC = {roc_auc:.3f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1, label='Random baseline')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig('Outputs/charts/09_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Outputs/charts/09_model_evaluation.png")

# ── Feature Importance Chart ─────────────────────────────
# Shows WHICH features the model used most to predict delays
# Only works for tree-based models (Random Forest, XGBoost)
if hasattr(model, 'feature_importances_'):
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Importance — what drives delay prediction',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('Outputs/charts/10_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: Outputs/charts/10_feature_importance.png")
    print(f"\nTop 3 most important features:")
    for feat, score in fi.sort_values(ascending=False).head(3).items():
        print(f"  {feat:20s}: {score:.4f}")

print("\nmodel_evaluation.py complete!")