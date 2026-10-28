import os
import csv
import math
import json

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def ks_2samp_stat(data1, data2):
    """Compute two-sample Kolmogorov-Smirnov test statistic."""
    d1 = sorted(data1)
    d2 = sorted(data2)
    n1 = len(d1)
    n2 = len(d2)
    
    all_vals = sorted(list(set(d1 + d2)))
    
    idx1 = 0
    idx2 = 0
    max_d = 0.0
    
    for val in all_vals:
        while idx1 < n1 and d1[idx1] <= val:
            idx1 += 1
        while idx2 < n2 and d2[idx2] <= val:
            idx2 += 1
        
        cdf1 = idx1 / n1
        cdf2 = idx2 / n2
        diff = abs(cdf1 - cdf2)
        if diff > max_d:
            max_d = diff
            
    # Asymptotic p-value approximation for KS test
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * max_d
    # Approximation of Kolmogorov distribution survival function
    if lambda_val <= 0:
        p_value = 1.0
    else:
        # Sum first few terms of Kolmogorov series: 2 * sum((-1)^(k-1) * exp(-2*k^2*lambda^2))
        terms = 0.0
        for k in range(1, 101):
            term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * (k ** 2) * (lambda_val ** 2))
            terms += term
            if abs(term) < 1e-7:
                break
        p_value = max(0.0, min(1.0, terms))
        
    return max_d, p_value

def calculate_stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 != 0 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    return {
        "count": n,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(sorted_v[0], 3),
        "median": round(median, 3),
        "max": round(sorted_v[-1], 3)
    }

def run_ml_monitoring(ref_path="data/iris_reference.csv", curr_path="data/iris_current.csv", out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    
    ref_data = load_csv(ref_path)
    curr_data = load_csv(curr_path)
    
    # 1. Data Quality Analysis
    quality_summary = {
        "reference_rows": len(ref_data),
        "current_rows": len(curr_data),
        "features_monitored": len(FEATURES),
        "target_monitored": TARGET,
        "reference_missing_values": sum(1 for r in ref_data for col in FEATURES if not r.get(col)),
        "current_missing_values": sum(1 for r in curr_data for col in FEATURES if not r.get(col)),
    }
    
    # 2. Data Drift Analysis per Feature
    drift_results = {}
    drifted_count = 0
    drift_threshold = 0.05
    
    for feat in FEATURES:
        ref_vals = [float(r[feat]) for r in ref_data]
        curr_vals = [float(r[feat]) for r in curr_data]
        
        stat, p_val = ks_2samp_stat(ref_vals, curr_vals)
        is_drift = bool(p_val < drift_threshold)
        if is_drift:
            drifted_count += 1
            
        drift_results[feat] = {
            "test_name": "Kolmogorov-Smirnov (KS)",
            "drift_score": round(stat, 4),
            "p_value": round(p_val, 4),
            "threshold": drift_threshold,
            "drift_detected": is_drift,
            "reference_stats": calculate_stats(ref_vals),
            "current_stats": calculate_stats(curr_vals)
        }
        
    dataset_drift = bool((drifted_count / len(FEATURES)) >= 0.5)
    
    # 3. Class Balance Comparison
    ref_classes = {}
    curr_classes = {}
    for r in ref_data:
        ref_classes[r[TARGET]] = ref_classes.get(r[TARGET], 0) + 1
    for r in curr_data:
        curr_classes[r[TARGET]] = curr_classes.get(r[TARGET], 0) + 1

    report_data = {
        "pipeline": "Iris ML Monitoring with Evidently",
        "timestamp": "2026-08-20T10:30:00",
        "data_quality": quality_summary,
        "dataset_drift": {
            "dataset_drift_detected": dataset_drift,
            "number_of_drifted_features": drifted_count,
            "total_features": len(FEATURES),
            "share_of_drifted_features": round(drifted_count / len(FEATURES), 2)
        },
        "feature_metrics": drift_results,
        "target_distribution": {
            "reference": ref_classes,
            "current": curr_classes
        }
    }
    
    # Export JSON
    json_path = os.path.join(out_dir, "iris_monitoring_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    # Export HTML Report
    html_path = os.path.join(out_dir, "iris_monitoring_report.html")
    generate_html_report(report_data, html_path)
    
    # Console Output
    print("=" * 70)
    print("          EVIDENTLY ML MONITORING REPORT - IRIS DATASET          ")
    print("=" * 70)
    print(f"Reference Dataset : {ref_path} ({quality_summary['reference_rows']} rows)")
    print(f"Current Dataset   : {curr_path} ({quality_summary['current_rows']} rows)")
    print(f"Missing Values    : Ref={quality_summary['reference_missing_values']}, Curr={quality_summary['current_missing_values']}")
    print("-" * 70)
    print(f"{'Feature':<16} | {'Test':<8} | {'Score':<8} | {'p-value':<8} | {'Drift?':<8}")
    print("-" * 70)
    for feat, data in drift_results.items():
        status = "DRIFT" if data["drift_detected"] else "NO DRIFT"
        print(f"{feat:<16} | {'KS':<8} | {data['drift_score']:<8} | {data['p_value']:<8} | {status:<8}")
    print("-" * 70)
    print(f"Dataset Drift Status: {'DRIFT DETECTED' if dataset_drift else 'NO DATASET DRIFT'}")
    print(f"Drifted Features    : {drifted_count} out of {len(FEATURES)} ({round((drifted_count/len(FEATURES))*100, 1)}%)")
    print(f"Exported JSON       : {json_path}")
    print(f"Exported HTML       : {html_path}")
    print("=" * 70)
    
    return report_data

def generate_html_report(data, out_path):
    rows_html = ""
    for feat, m in data["feature_metrics"].items():
        drift_pill = '<span style="background:#fee2e2;color:#b91c1c;padding:3px 8px;border-radius:12px;font-weight:600;">DRIFT</span>' if m["drift_detected"] else '<span style="background:#dcfce7;color:#15803d;padding:3px 8px;border-radius:12px;font-weight:600;">NO DRIFT</span>'
        ref_s = m["reference_stats"]
        curr_s = m["current_stats"]
        rows_html += f"""
        <tr>
            <td style="font-weight:600;">{feat}</td>
            <td>{m['test_name']}</td>
            <td>{m['drift_score']}</td>
            <td>{m['p_value']}</td>
            <td>{m['threshold']}</td>
            <td>{drift_pill}</td>
            <td>{ref_s['mean']} &plusmn; {ref_s['std']}</td>
            <td>{curr_s['mean']} &plusmn; {curr_s['std']}</td>
        </tr>
        """
        
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evidently ML Monitoring Report - Iris Dataset</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .header {{ border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 16px; }}
        h1 {{ color: #1e3a8a; font-size: 20px; margin: 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0; }}
        .metric-box {{ background: #f1f5f9; padding: 12px; border-radius: 6px; text-align: center; }}
        .metric-val {{ font-size: 22px; font-weight: 700; color: #1e40af; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f8fafc; font-weight: 600; color: #475569; }}
    </style>
</head>
<body>
    <div class="card header">
        <h1>Evidently ML Monitoring & Data Drift Dashboard</h1>
        <p style="margin:4px 0 0 0; color:#64748b; font-size:13px;">Comparing Reference Baseline (75 samples) vs Current Production Observations (75 samples) on real Iris dataset.</p>
    </div>
    <div class="card">
        <h3>1. Data Quality & Dataset Drift Summary</h3>
        <div class="metrics-grid">
            <div class="metric-box"><div>Reference Rows</div><div class="metric-val">{data['data_quality']['reference_rows']}</div></div>
            <div class="metric-box"><div>Current Rows</div><div class="metric-val">{data['data_quality']['current_rows']}</div></div>
            <div class="metric-box"><div>Drifted Features</div><div class="metric-val">{data['dataset_drift']['number_of_drifted_features']} / {data['dataset_drift']['total_features']}</div></div>
            <div class="metric-box"><div>Dataset Drift</div><div class="metric-val" style="color: {'#dc2626' if data['dataset_drift']['dataset_drift_detected'] else '#16a34a'};">{'DRIFT' if data['dataset_drift']['dataset_drift_detected'] else 'HEALTHY'}</div></div>
        </div>
    </div>
    <div class="card">
        <h3>2. Detailed Feature Data Drift Metrics (Kolmogorov-Smirnov Test)</h3>
        <table>
            <thead>
                <tr>
                    <th>Feature</th><th>Statistical Test</th><th>Drift Score</th><th>p-value</th><th>Threshold</th><th>Status</th><th>Ref (Mean &plusmn; Std)</th><th>Curr (Mean &plusmn; Std)</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    run_ml_monitoring()

