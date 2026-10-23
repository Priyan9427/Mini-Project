import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import calculate_metrics
from representation_generator import apply_low_resolution, isolate_extremes, apply_domain_shift

st.set_page_config(page_title="Hybrid AI Auditor", page_icon="🛡️", layout="wide")

st.title("🛡️ Hybrid AI & Statistical Hypothesis Auditor")
st.markdown("""
This Dual-Engine framework subjects your dataset to **Multiverse Attacks**. It uses **CUDA-Accelerated XGBoost** 
to map non-linear cognitive drift, and **Classical OLS Statistics** to guarantee strict P-Value confidence.
""")

with st.sidebar:
    st.header("📂 1. Data Ingestion")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        # .dropna() removes broken sensor rows so classical statistics don't crash
        df = pd.read_csv(uploaded_file).dropna()
        st.success(f"Dataset loaded and cleaned. Ready for audit: {len(df)} valid rows.")
        
        st.header("🎯 2. Hypothesis Mapping")
        columns = df.columns.tolist()
        target = st.selectbox("Select Target (What are you predicting?)", columns)
        
        feature_options = [c for c in columns if c != target]
        features = st.multiselect("Select Inputs (Your sensors/variables)", feature_options)

if uploaded_file and target and len(features) > 0:
    st.divider()
    
    if st.button("🚀 Run Dual-Engine Audit", type="primary", use_container_width=True):
        with st.spinner("Waking up RTX GPU... Running AI and Statistical Engines..."):
            
            audit_results = {}
            
            # --- EXECUTE MULTIVERSE ATTACKS ---
            # 1. Baseline: The AI learns the standard relationship
            audit_results["Baseline"] = calculate_metrics(df, features, target)
            
            baseline_imp = audit_results['Baseline']['importance']
            primary_feature = max(baseline_imp, key=baseline_imp.get)
            
            # 2. Low-Res Sensors: Testing robustness against fuzzy data
            low_res_df = apply_low_resolution(df, features)
            audit_results["Low-Res Sensors"] = calculate_metrics(low_res_df, features, target)
            
            # 3. Extreme Anomalies: Testing physical limits (e.g., deep ocean)
            extreme_df = isolate_extremes(df, target)
            audit_results["Extreme Anomalies"] = calculate_metrics(extreme_df, features, target)
            
            # 4. Domain Shift: Testing systemic bias
            domain_df = apply_domain_shift(df, primary_feature)
            audit_results["Domain Shift (High)"] = calculate_metrics(domain_df, features, target)
            
            # --- EVALUATE HYBRID SAFETY RULES (UNIVERSAL DIMENSIONLESS MODE) ---
            baseline_r2 = audit_results['Baseline']['r2']
            is_stable = True
            failure_reasons = []
            
            for test_name, metrics in audit_results.items():
                if test_name == 'Baseline': continue
                
                # Rule 1: Phase Shift vs Cognitive Drift (Checks the SHAP Anchor)
                # If the AI changes its logic but accuracy collapses, it is broken (Drift).
                # If it changes its logic but maintains accuracy, it discovered a real physical adaptation (Phase Shift).
                current_primary = max(metrics['importance'], key=metrics['importance'].get)
                if current_primary != primary_feature:
                    if metrics['r2'] < 0.90:
                        failure_reasons.append(f"**{test_name}**: Cognitive Drift! AI abandoned '{primary_feature}' for '{current_primary}'.")
                        is_stable = False
                    else:
                        st.sidebar.success(f"🌊 **Phase Shift Detected ({test_name})**: AI safely adapted to use '{current_primary}'.")
                
                # Rule 2: Universal Model Stability (Checks Dimensionless R-Squared)
                # This works for ANY hypothesis. If accuracy collapses by > 25%, the mathematical relationship broke.
                r2_drop = ((baseline_r2 - metrics['r2']) / baseline_r2) * 100
                if r2_drop > 25.0:
                    failure_reasons.append(f"**{test_name}**: R-Squared collapsed by {r2_drop:.1f}%. The mathematical relationship broke.")
                    is_stable = False
                    
                # NOTE: The RMSE rule has been entirely removed to ensure the system 
                # is strictly scale-invariant and reliant purely on dimensionless R2 metrics.
                    
                # Rule 3: Statistical Confidence Limit (Checks Dimensionless P-Value)
                # A P-Value > 0.05 means classical statistics cannot prove the AI didn't just get lucky.
                if metrics['p_value'] > 0.05:
                    failure_reasons.append(f"**{test_name}**: Statistical Failure! P-Value reached {metrics['p_value']:.4f}, breaking the 0.05 confidence limit.")
                    is_stable = False

            # --- RENDER THE DASHBOARD ---
            st.header("📊 Hybrid Audit Results")
            
            if is_stable:
                st.success("✅ FINAL VERDICT: 1 (SAFE & STABLE) - The hypothesis survived all AI and Statistical tests.")
            else:
                st.error("❌ FINAL VERDICT: 0 (UNSAFE) - The hypothesis failed the Audit.")
                with st.expander("View Critical Failure Logs", expanded=True):
                    for reason in failure_reasons:
                        st.markdown(f"- {reason}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**SHAP Baseline Cognitive Anchor**")
                st.info(f"🧠 The AI relies mostly on: **{primary_feature}** ({baseline_imp[primary_feature]:.1f}%)")
                
                labels = list(audit_results.keys())
                scores = [data['r2'] for data in audit_results.values()]
                
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0']
                bars = ax.bar(labels, scores, color=colors[:len(labels)])
                ax.set_ylim(0.0, 1.1)
                ax.set_ylabel('R-Squared Score')
                ax.set_title('Model AI Stability Across Multiverse Attacks')
                st.pyplot(fig)
                
            with col2:
                st.markdown("**Detailed Perturbation Metrics**")
                
                table_data = []
                for test, metrics in audit_results.items():
                    anchor = max(metrics['importance'], key=metrics['importance'].get)
                    
                    # Formatting P-Value: Uses scientific notation (e.g., 1.4e-15) for microscopic numbers 
                    # so they don't visually round to a misleading 0.0000.
                    raw_p = metrics['p_value']
                    formatted_p = f"{raw_p:.2e}" if raw_p < 0.0001 else round(raw_p, 4)
                    
                    table_data.append({
                        "Attack Vector": test,
                        "R-Squared": round(metrics['r2'], 4),
                        "RMSE": round(metrics['rmse'], 4), # We still display RMSE so the scientist can see the physical error, we just don't use it for pass/fail.
                        "P-Value": formatted_p, 
                        "Cognitive Anchor": anchor
                    })
                    
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

elif not uploaded_file:
    st.info("👈 Please upload a CSV file in the sidebar to begin.")