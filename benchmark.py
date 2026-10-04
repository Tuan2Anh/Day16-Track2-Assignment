import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

def main():
    print("=" * 60)
    print("BẮT ĐẦU BENCHMARK LIGHTGBM TRÊN CREDIT CARD FRAUD")
    print("=" * 60)

    # 1. Load Data
    data_path = "/home/ubuntu/ml-benchmark/creditcard.csv"
    print(f"\n[1/5] Đang load dữ liệu từ {data_path}...")
    t0 = time.time()
    df = pd.read_csv(data_path)
    data_load_time = round(time.time() - t0, 4)
    print(f"-> Đã load {df.shape[0]:,} dòng, {df.shape[1]} cột trong {data_load_time} giây.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Train/Test Split
    print("\n[2/5] Tách tập dữ liệu Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"-> Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    # 3. Train LightGBM Model
    print("\n[3/5] Đang huấn luyện LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    t_train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    training_time = round(time.time() - t_train_start, 4)
    best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") and model.best_iteration_ else 100
    print(f"-> Training hoàn tất trong {training_time} giây! Best iteration: {best_iter}")

    # 4. Đánh giá Model trên tập Test
    print("\n[4/5] Đánh giá mô hình trên tập Test...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_roc = round(float(roc_auc_score(y_test, y_pred_proba)), 4)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)

    # 5. Đo Inference Latency & Throughput
    print("\n[5/5] Đo lường Inference Latency và Throughput...")
    # Single row latency (lấy trung bình qua 100 lần dự đoán)
    sample_single = X_test.iloc[[0]]
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        _ = model.predict_proba(sample_single)
        latencies.append((time.perf_counter() - t_start) * 1000) # milliseconds
    inf_latency_ms = round(float(np.mean(latencies)), 3)

    # Batch throughput (1000 rows)
    sample_1000 = X_test.iloc[:1000]
    t_start = time.perf_counter()
    _ = model.predict_proba(sample_1000)
    batch_time = time.perf_counter() - t_start
    inf_throughput = round(float(len(sample_1000) / batch_time), 1) # rows/sec

    # Kết quả tổng hợp
    results = {
        "load_time_sec": data_load_time,
        "training_time_sec": training_time,
        "best_iteration": int(best_iter),
        "auc_roc": auc_roc,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "inference_latency_ms": inf_latency_ms,
        "inference_throughput_rows_per_sec": inf_throughput
    }

    # Xuất ra file JSON
    output_path = "/home/ubuntu/ml-benchmark/benchmark_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n-> Đã lưu kết quả vào: {output_path}")

    # In bảng kết quả
    print("\n" + "=" * 60)
    print("BẢNG KẾT QUẢ BENCHMARK (ĐIỀN VÀO BÁO CÁO NỘP BÀI)")
    print("=" * 60)
    print(f"| {'Metric':<35} | {'Kết quả':<18} |")
    print("|" + "-" * 37 + "|" + "-" * 20 + "|")
    print(f"| {'Thời gian load data':<35} | {str(data_load_time) + ' s':<18} |")
    print(f"| {'Thời gian training':<35} | {str(training_time) + ' s':<18} |")
    print(f"| {'Best iteration':<35} | {str(best_iter):<18} |")
    print(f"| {'AUC-ROC':<35} | {str(auc_roc):<18} |")
    print(f"| {'Accuracy':<35} | {str(accuracy):<18} |")
    print(f"| {'F1-Score':<35} | {str(f1):<18} |")
    print(f"| {'Precision':<35} | {str(precision):<18} |")
    print(f"| {'Recall':<35} | {str(recall):<18} |")
    print(f"| {'Inference latency (1 row)':<35} | {str(inf_latency_ms) + ' ms':<18} |")
    print(f"| {'Inference throughput (1000 rows)':<35} | {str(inf_throughput) + ' rows/s':<18} |")
    print("=" * 60)

if __name__ == "__main__":
    main()
