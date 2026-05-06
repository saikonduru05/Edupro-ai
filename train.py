import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("🚀 Training started")

df = pd.read_csv("data/transactions.csv")
df.columns = df.columns.str.strip().str.lower()

df = df.drop(["transactionid", "transactiondate"], axis=1, errors="ignore")

# ---------------- REDUCE HIGH DIMENSION ----------------
df["course_teacher"] = df["courseid"].astype(str) + "_" + df["teacherid"].astype(str)

# 🔥 KEEP ONLY IMPORTANT NUMERIC SIGNALS
df["course_code"] = df["courseid"].astype("category").cat.codes
df["teacher_code"] = df["teacherid"].astype("category").cat.codes
df["user_code"] = df["userid"].astype("category").cat.codes
df["payment_code"] = df["paymentmethod"].astype("category").cat.codes

# TARGET
y = df["amount"]

# FEATURES (NO ONE-HOT NOW → BIG FIX)
X = df[["course_code", "teacher_code", "user_code", "payment_code"]]

print("📦 Reduced Feature shape:", X.shape)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL (better for numeric features)
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": model,
    "columns": X.columns
}, "models/best_model.pkl")

print("✅ TRAINING COMPLETED (OPTIMIZED MODEL)")