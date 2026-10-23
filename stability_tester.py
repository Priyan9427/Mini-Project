import pandas as pd
from analyzer import calculate_metrics
from representation_generator import apply_low_resolution, isolate_extremes, apply_domain_shift
from profile_builder import generate_dashboard

def calculate_final_verdict(audit_results, max_r2_drop=25.0, max_rmse_spike=50.0):
    print("\n⚖️ Running AI Safety Heuristic Classifier...")
    baseline_r2 = audit_results['Baseline']['r2']
    baseline_rmse = audit_results['Baseline']['rmse']
    
    baseline_importance = audit_results['Baseline']['importance']
    primary_feature = max(baseline_importance, key=baseline_importance.get)
    print(f"🧠 Baseline Cognitive Anchor: AI relies mostly on '{primary_feature}' ({baseline_importance[primary_feature]:.1f}%)")
    
    is_stable = True 
    
    for test_name, metrics in audit_results.items():
        if test_name == 'Baseline':
            continue
            
        # RULE 1: Cognitive Drift (SHAP XAI)
        current_importance = metrics['importance']
        current_primary = max(current_importance, key=current_importance.get)
        
        if current_primary != primary_feature:
            print(f"   ❌ FAILED: '{test_name}' caused Cognitive Drift! Model shifted reliance from {primary_feature} to {current_primary}.")
            is_stable = False
            
        # RULE 2: R-Squared Variance (Expanded to 25% because Extreme Anomaly tests are very harsh)
        r2_drop_pct = ((baseline_r2 - metrics['r2']) / baseline_r2) * 100
        if r2_drop_pct > max_r2_drop:
            print(f"   ❌ FAILED: '{test_name}' caused R-squared to drop by {r2_drop_pct:.2f}%.")
            is_stable = False
            
        # RULE 3: RMSE Spike
        rmse_spike_pct = ((metrics['rmse'] - baseline_rmse) / baseline_rmse) * 100
        if rmse_spike_pct > max_rmse_spike:
            print(f"   ❌ FAILED: '{test_name}' caused Physical Error to spike by {rmse_spike_pct:.2f}%.")
            is_stable = False
            
    final_score = 1 if is_stable else 0
    print(f"\n🎯 FINAL SYSTEM VERDICT: {final_score}")
    return final_score

def run_audit():
    print("===================================================")
    print("  Universal Representation-Aware Hypothesis Auditor")
    print("===================================================\n")
    
    file_path = input("📂 Enter the dataset filename (e.g., ocean_data.csv): ")
    target = input("🎯 Enter the Target column to predict (e.g., density): ")
    features_input = input("🔢 Enter the Input columns separated by commas (e.g., temperature, depth, salinity): ")
    features = [f.strip() for f in features_input.split(',')]
    
    print("\n🚀 Initiating Multiverse Audit Pipeline...\n")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"\n❌ Error: The file '{file_path}' was not found.")
        return
    
    audit_results = {}
    
    # 1. Baseline
    audit_results["Baseline"] = calculate_metrics(df, features, target)
    
    # We dynamically identify the primary feature to use in the Domain Shift attack
    primary_feature = max(audit_results['Baseline']['importance'], key=audit_results['Baseline']['importance'].get)
    
    # 2. Low-Resolution Sensor Attack
    low_res_df = apply_low_resolution(df, features)
    audit_results["Low-Res Sensors"] = calculate_metrics(low_res_df, features, target)
    
    # 3. Extreme Anomaly Attack
    extreme_df = isolate_extremes(df, target)
    audit_results["Extreme Anomalies"] = calculate_metrics(extreme_df, features, target)
    
    # 4. Domain Shift Attack
    domain_df = apply_domain_shift(df, primary_feature)
    audit_results["Domain Shift (High)"] = calculate_metrics(domain_df, features, target)
    
    print("✅ All math calculated. Evaluating AI Safety thresholds...")
    verdict = calculate_final_verdict(audit_results)
    
    generate_dashboard(audit_results, verdict)

if __name__ == "__main__":
    run_audit()