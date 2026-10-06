import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import joblib
from scipy.stats import ks_2samp

# ML & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier

# Visualizations
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Multi-Dataset MLOps Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0d6efd;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BUILT-IN DATASET GENERATORS
# ==========================================
@st.cache_data
def load_flights_dataset():
    """Generates Flight Price Dataset (flights.csv schema)."""
    np.random.seed(42)
    n_samples = 1500
    
    agencies = ["FlyingDrops", "CloudFy", "Rainbow"]
    flight_types = ["economic", "premium", "firstClass"]
    cities = ["Recife (PE)", "Florianopolis (SC)", "Brasilia (DF)", "Aracaju (SE)", "Salvador (BH)", "Sao Paulo (SP)", "Natal (RN)"]
    
    travel_code = np.random.randint(0, 500, size=n_samples)
    user_code = np.random.randint(0, 100, size=n_samples)
    
    from_city = np.random.choice(cities, size=n_samples)
    to_city = np.random.choice(cities, size=n_samples)
    to_city = np.where(to_city == from_city, "Sao Paulo (SP)", to_city)
    
    flight_type = np.random.choice(flight_types, size=n_samples, p=[0.5, 0.3, 0.2])
    agency = np.random.choice(agencies, size=n_samples)
    
    flight_time = np.round(np.random.uniform(0.5, 3.5, size=n_samples), 2)
    distance = np.round(flight_time * np.random.uniform(350, 420, size=n_samples), 2)
    
    base_price = distance * 1.2 + flight_time * 50
    class_multiplier = np.where(flight_type == "firstClass", 2.2, np.where(flight_type == "premium", 1.5, 1.0))
    agency_multiplier = np.where(agency == "FlyingDrops", 1.1, 1.0)
    
    price = np.round(base_price * class_multiplier * agency_multiplier + np.random.normal(0, 40, size=n_samples), 2)
    price = np.where(price < 100, 150.0, price)
    
    dates = pd.date_range(start="2019-09-01", periods=n_samples, freq="h").strftime("%m/%d/%Y")

    return pd.DataFrame({
        "travelCode": travel_code,
        "userCode": user_code,
        "from": from_city,
        "to": to_city,
        "flightType": flight_type,
        "price": price,
        "time": flight_time,
        "distance": distance,
        "agency": agency,
        "date": dates
    })

@st.cache_data
def load_churn_dataset():
    """Generates Customer Churn Dataset."""
    np.random.seed(42)
    n_samples = 1000
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, size=n_samples), 2)
    total_charges = np.where(total_charges < 0, 0, total_charges)
    support_calls = np.random.poisson(lam=2, size=n_samples)
    
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.4, 0.2])
    payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)

    churn_prob = (
        0.3 * (contract_type == "Month-to-month") +
        0.2 * (support_calls > 3) +
        0.2 * (monthly_charges > 80) -
        0.3 * (tenure > 24)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    return pd.DataFrame({
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportCalls": support_calls,
        "Gender": gender,
        "ContractType": contract_type,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "Churn": churn
    })

@st.cache_data
def load_housing_dataset():
    """Generates Housing Price Dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    square_feet = np.random.randint(600, 4500, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.randint(1, 4, size=n_samples)
    year_built = np.random.randint(1970, 2023, size=n_samples)
    garage_cars = np.random.randint(0, 4, size=n_samples)
    neighborhood = np.random.choice(["Suburbs", "Downtown", "Urban", "Rural"], size=n_samples)
    condition = np.random.choice(["Fair", "Good", "Excellent"], size=n_samples)
    
    location_mult = np.where(neighborhood == "Downtown", 1.4, np.where(neighborhood == "Suburbs", 1.2, 1.0))
    sale_price = (square_feet * 120 + bedrooms * 15000 + bathrooms * 20000 + garage_cars * 10000) * location_mult
    sale_price = np.round(sale_price + np.random.normal(0, 25000, size=n_samples), -2)

    return pd.DataFrame({
        "SquareFeet": square_feet,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "Neighborhood": neighborhood,
        "Condition": condition,
        "SalePrice": sale_price
    })

def calculate_psi(baseline, target, num_buckets=10):
    """Calculates Population Stability Index (PSI)."""
    try:
        baseline = baseline.dropna()
        target = target.dropna()
        if len(baseline) == 0 or len(target) == 0:
            return 0.0
            
        quantiles = np.linspace(0, 1, num_buckets + 1)
        buckets = np.percentile(baseline, quantiles * 100)
        buckets[0] = -np.inf
        buckets[-1] = np.inf

        baseline_counts = np.histogram(baseline, bins=buckets)[0]
        target_counts = np.histogram(target, bins=buckets)[0]

        baseline_pct = np.where(baseline_counts == 0, 0.0001, baseline_counts) / len(baseline)
        target_pct = np.where(target_counts == 0, 0.0001, target_counts) / len(target)

        psi_val = np.sum((target_pct - baseline_pct) * np.log(target_pct / baseline_pct))
        return round(float(psi_val), 4)
    except Exception:
        return 0.0

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Flights Price Dataset"
if "dataset" not in st.session_state:
    st.session_state.dataset = load_flights_dataset()
    st.session_state.target_col = "price"
    st.session_state.problem_type = "Regression"

if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "model_registry" not in st.session_state:
    st.session_state.model_registry = []
if "production_model" not in st.session_state:
    st.session_state.production_model = None
if "production_pipeline" not in st.session_state:
    st.session_state.production_pipeline = None

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🚀 MLOps Platform")
st.sidebar.caption("Multi-Dataset Intelligent Deployment")

menu_option = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home Dashboard",
        "📊 Dataset Management",
        "🔧 Preprocessing Pipeline",
        "🤖 Automated Model Training",
        "🧪 MLflow Experiments",
        "🏆 Model Registry",
        "🔮 Prediction Engine",
        "📈 Performance Monitoring",
        "🚨 Data Drift Detection"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"📌 **Dataset**: {st.session_state.dataset_name}\n\n**Target**: `{st.session_state.target_col}` ({st.session_state.problem_type})")

# ==========================================
# MODULE 1: HOME DASHBOARD
# ==========================================
if menu_option == "🏠 Home Dashboard":
    st.title("🏠 Intelligent MLOps Platform Overview")
    st.markdown("End-to-end Machine Learning deployment and monitoring platform with built-in multi-dataset support.")

    # KPI Summary Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Dataset", st.session_state.dataset_name.split()[0])
    m2.metric("Dataset Rows", f"{len(st.session_state.dataset):,}" if st.session_state.dataset is not None else "0")
    m3.metric("Experiments Run", len(st.session_state.experiments))
    
    prod_model_name = "None"
    prod_metric = "N/A"
    if st.session_state.production_model:
        prod_model_name = st.session_state.production_model.get("model_name", "Deployed")
        metrics = st.session_state.production_model.get('metrics', {})
        score = metrics.get('R2', metrics.get('Accuracy', 0))
        prod_metric = f"{score:.3f}"
        
    m4.metric("Production Model", prod_model_name)
    m5.metric("System Health", "Healthy 🟢")

    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Model Performance Benchmark")
        if len(st.session_state.experiments) > 0:
            exp_df = pd.DataFrame(st.session_state.experiments)
            metric_col = "R2" if "R2" in exp_df.columns else "Accuracy"
            fig = px.bar(
                exp_df, 
                x="Model", 
                y=metric_col, 
                color="Model",
                title=f"Experiment Benchmark ({metric_col})",
                text_auto='.3f'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No experiment runs available yet. Go to 'Automated Model Training' to run model benchmarks.")
            
    with col2:
        st.subheader("🏆 Active Model Status")
        if st.session_state.production_model:
            st.success(f"**Version**: {st.session_state.production_model.get('version', 'v1')}")
            st.write(f"**Algorithm**: {st.session_state.production_model.get('model_name')}")
            st.write(f"**Promoted On**: {st.session_state.production_model.get('date')}")
            st.write("**Stage**: Production 🟢")
        else:
            st.warning("No model promoted to production yet.")

# ==========================================
# MODULE 2: DATASET MANAGEMENT
# ==========================================
elif menu_option == "📊 Dataset Management":
    st.title("📊 Dataset Management & Exploratory Analysis")
    
    st.subheader("Select Dataset Source")
    dataset_choice = st.selectbox(
        "Choose Built-in Dataset or Upload Custom CSV",
        [
            "✈️ Flights Price Dataset (flights.csv)",
            "👥 Customer Churn Dataset",
            "🏠 Housing Price Dataset",
            "📁 Upload Custom CSV File..."
        ]
    )

    if dataset_choice == "✈️ Flights Price Dataset (flights.csv)":
        st.session_state.dataset_name = "Flights Price Dataset"
        st.session_state.dataset = load_flights_dataset()
        st.session_state.target_col = "price"
        st.session_state.problem_type = "Regression"

    elif dataset_choice == "👥 Customer Churn Dataset":
        st.session_state.dataset_name = "Customer Churn Dataset"
        st.session_state.dataset = load_churn_dataset()
        st.session_state.target_col = "Churn"
        st.session_state.problem_type = "Classification"

    elif dataset_choice == "🏠 Housing Price Dataset":
        st.session_state.dataset_name = "Housing Price Dataset"
        st.session_state.dataset = load_housing_dataset()
        st.session_state.target_col = "SalePrice"
        st.session_state.problem_type = "Regression"

    elif dataset_choice == "📁 Upload Custom CSV File...":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.dataset = pd.read_csv(uploaded_file)
                st.session_state.dataset_name = uploaded_file.name
                st.success("Custom CSV uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    df = st.session_state.dataset
    if df is not None:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isna().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())

        st.subheader(f"Dataset Preview: {st.session_state.dataset_name}")
        st.dataframe(df.head(10), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📋 Summary Statistics", "🎯 Target Distribution", "🔥 Correlation Matrix"])
        
        with tab1:
            st.write("**Numerical Summary**")
            st.dataframe(df.describe().T, use_container_width=True)

        with tab2:
            target_candidate = st.selectbox(
                "Select Target Column", 
                df.columns, 
                index=df.columns.get_loc(st.session_state.target_col) if st.session_state.target_col in df.columns else len(df.columns)-1
            )
            st.session_state.target_col = target_candidate
            
            if df[target_candidate].nunique() <= 10:
                st.session_state.problem_type = "Classification"
                fig = px.pie(df, names=target_candidate, title=f"Class Distribution: {target_candidate}")
            else:
                st.session_state.problem_type = "Regression"
                fig = px.histogram(df, x=target_candidate, title=f"Target Value Distribution ({target_candidate})", marginal="box", color_discrete_sequence=['#0d6efd'])
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Numerical Features Heatmap", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 3: PREPROCESSING PIPELINE
# ==========================================
elif menu_option == "🔧 Preprocessing Pipeline":
    st.title("🔧 Data Preprocessing & Column Transformer")
    df = st.session_state.dataset

    if df is None:
        st.warning("Please select a dataset first.")
    else:
        target_col = st.session_state.target_col
        st.write(f"Active Dataset: **{st.session_state.dataset_name}**")
        st.write(f"Target Column: **{target_col}** | Type: **{st.session_state.problem_type}**")
        
        feature_cols = [c for c in df.columns if c not in [target_col, "travelCode", "userCode", "date"]]
        X = df[feature_cols]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Numerical Features")
            st.write(f"Detected ({len(num_cols)}): `{num_cols}`")
            num_imputer = st.selectbox("Imputer Strategy", ["median", "mean"])
            scaler_type = st.selectbox("Scaling Strategy", ["StandardScaler", "MinMaxScaler"])

        with c2:
            st.subheader("Categorical Features")
            st.write(f"Detected ({len(cat_cols)}): `{cat_cols}`")
            cat_imputer = st.selectbox("Imputer Strategy", ["most_frequent"])
            cat_encoder = st.selectbox("Encoding Strategy", ["OneHotEncoder"])

        st.subheader("🧹 Dynamic Outlier Filtering Controls")
        iqr_threshold = st.slider("IQR Outlier Removal Threshold Multiplier", 1.0, 3.0, 1.5, 0.1)

        if st.button("⚙️ Compile Scikit-Learn Pipeline"):
            st.success("ColumnTransformer and Feature Preprocessing Pipeline successfully built!")

# ==========================================
# MODULE 4 & 5: MODEL TRAINING & TRACKING
# ==========================================
elif menu_option == "🤖 Automated Model Training":
    st.title("🤖 Automated Model Training & Benchmark")
    df = st.session_state.dataset

    if df is None:
        st.warning("Please select a dataset first.")
    else:
        target_col = st.session_state.target_col
        feature_cols = [c for c in df.columns if c not in [target_col, "travelCode", "userCode", "date"]]
        
        X = df[feature_cols]
        y = df[target_col]

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        test_size = st.slider("Train/Test Split Ratio", 0.1, 0.4, 0.2)

        # 🎛️ Interactive Hyperparameter Tuning Controls
        st.subheader("🎛️ Hyperparameter Configuration")
        with st.expander("Configure Algorithm Hyperparameters"):
            rf_trees = st.slider("Random Forest - Number of Trees (n_estimators)", 10, 200, 100, 10)
            rf_depth = st.slider("Random Forest - Max Depth", 2, 20, 10)
            xgb_lr = st.select_slider("XGBoost - Learning Rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)

        if st.session_state.problem_type == "Regression":
            st.subheader("Select Regression Algorithms")
            run_lr = st.checkbox("Linear Regression", value=True)
            run_dt = st.checkbox("Decision Tree Regressor", value=True)
            run_rf = st.checkbox("Random Forest Regressor", value=True)
            run_xgb = st.checkbox("XGBoost Regressor", value=True)
        else:
            st.subheader("Select Classification Algorithms")
            run_lr = st.checkbox("Logistic Regression", value=True)
            run_rf = st.checkbox("Random Forest Classifier", value=True)
            run_xgb = st.checkbox("XGBoost Classifier", value=True)
            run_knn = st.checkbox("K-Nearest Neighbors", value=True)
            run_svm = st.checkbox("Support Vector Machine (SVM)", value=True)

        if st.button("🚀 Train All Selected Algorithms"):
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, num_cols),
                    ('cat', cat_transformer, cat_cols)
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            models = {}
            if st.session_state.problem_type == "Regression":
                if run_lr: models["Linear Regression"] = LinearRegression()
                if run_dt: models["Decision Tree"] = DecisionTreeRegressor(max_depth=rf_depth)
                if run_rf: models["Random Forest"] = RandomForestRegressor(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
                if run_xgb: models["XGBoost"] = XGBRegressor(learning_rate=xgb_lr, random_state=42)
            else:
                if run_lr: models["Logistic Regression"] = LogisticRegression()
                if run_rf: models["Random Forest"] = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
                if run_xgb: models["XGBoost"] = XGBClassifier(learning_rate=xgb_lr, eval_metric='logloss', random_state=42)
                if 'run_knn' in locals() and run_knn: models["KNN"] = KNeighborsClassifier()
                if 'run_svm' in locals() and run_svm: models["SVM"] = SVC(probability=True)

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            idx = 0
            for name, clf in models.items():
                idx += 1
                status_text.text(f"Training model {idx}/{len(models)}: {name}...")
                
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
                
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                elapsed_time = round(time.time() - start_time, 3)

                y_pred = pipeline.predict(X_test)

                if st.session_state.problem_type == "Regression":
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    metrics = {"MAE": round(mae, 2), "MSE": round(mse, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
                else:
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics = {"Accuracy": round(acc, 4), "F1": round(f1, 4)}

                experiment_record = {
                    "Experiment_ID": f"EXP-{len(st.session_state.experiments) + 1:03d}",
                    "Dataset": st.session_state.dataset_name,
                    "Model": name,
                    "Training_Time_Sec": elapsed_time,
                    "Pipeline_Artifact": pipeline,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    **metrics
                }
                st.session_state.experiments.append(experiment_record)
                progress_bar.progress(idx / len(models))

            status_text.text("Training completed!")
            st.success("Models trained and logged to MLflow Experiment Tracker!")

        if len(st.session_state.experiments) > 0:
            st.markdown("---")
            st.subheader("📋 Experiment Results Comparison")
            results_df = pd.DataFrame(st.session_state.experiments).drop(columns=["Pipeline_Artifact", "y_test", "y_pred"], errors="ignore")
            st.dataframe(results_df.style.highlight_max(axis=0, color="#d1e7dd"), use_container_width=True)

            # 📈 Interactive Plotly Prediction Scatter Evaluation Plot
            st.subheader("📈 Interactive Prediction Evaluation Plot")
            latest_exp = st.session_state.experiments[-1]
            if st.session_state.problem_type == "Regression":
                fig_res = px.scatter(
                    x=latest_exp["y_test"], 
                    y=latest_exp["y_pred"], 
                    labels={'x': 'Actual Target Values', 'y': 'Predicted Target Values'},
                    title=f"Actual vs Predicted Scatter ({latest_exp['Model']})",
                    trendline="ols"
                )
                st.plotly_chart(fig_res, use_container_width=True)

# ==========================================
# MODULE 5: EXPERIMENTS STORE
# ==========================================
elif menu_option == "🧪 MLflow Experiments":
    st.title("🧪 MLflow Experiment Tracking Store")
    if len(st.session_state.experiments) == 0:
        st.info("No training runs found yet. Go to 'Automated Model Training' to run model benchmarks.")
    else:
        exp_df = pd.DataFrame(st.session_state.experiments).drop(columns=["Pipeline_Artifact", "y_test", "y_pred"], errors="ignore")
        st.dataframe(exp_df, use_container_width=True)

# ==========================================
# MODULE 7: MODEL REGISTRY & STAGE PROMOTION
# ==========================================
elif menu_option == "🏆 Model Registry":
    st.title("🏆 MLflow Model Registry")
    
    if len(st.session_state.experiments) == 0:
        st.warning("Train models before using the registry.")
    else:
        exp_df = pd.DataFrame(st.session_state.experiments)
        selected_exp_id = st.selectbox("Select Candidate Run to Promote", exp_df["Experiment_ID"])
        target_stage = st.selectbox("Target Stage", ["Production", "Staging", "Development", "Archived"])

        if st.button("🚀 Promote Stage"):
            exp_detail = next(item for item in st.session_state.experiments if item["Experiment_ID"] == selected_exp_id)
            
            registry_entry = {
                "version": f"v{len(st.session_state.model_registry) + 1}",
                "model_name": exp_detail["Model"],
                "stage": target_stage,
                "metrics": {k: exp_detail[k] for k in ["R2", "MAE", "Accuracy"] if k in exp_detail},
                "pipeline": exp_detail["Pipeline_Artifact"],
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.model_registry.append(registry_entry)
            
            if target_stage == "Production":
                st.session_state.production_model = registry_entry
                st.session_state.production_pipeline = exp_detail["Pipeline_Artifact"]
                st.success(f"Promoted **{exp_detail['Model']}** to PRODUCTION!")

        st.markdown("---")
        st.subheader("📋 Registry Log")
        if len(st.session_state.model_registry) > 0:
            reg_table = [{"Version": r["version"], "Model": r["model_name"], "Stage": r["stage"], "Date": r["date"]} for r in st.session_state.model_registry]
            st.dataframe(pd.DataFrame(reg_table), use_container_width=True)

        # 💾 Download Active Production Model (.joblib)
        if st.session_state.production_pipeline is not None:
            st.markdown("---")
            st.subheader("💾 Export Model Binary")
            buffer = io.BytesIO()
            joblib.dump(st.session_state.production_pipeline, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="💾 Download Active Production Model (.joblib)",
                data=buffer,
                file_name="production_model_pipeline.joblib",
                mime="application/octet-stream"
            )

# ==========================================
# MODULE 8: PREDICTION ENGINE
# ==========================================
elif menu_option == "🔮 Prediction Engine":
    st.title("🔮 Real-Time Prediction Engine")
    
    if st.session_state.production_pipeline is None:
        st.error("No Production model deployed! Please promote a model under 'Model Registry' first.")
    else:
        st.success(f"Serving Model: **{st.session_state.production_model['model_name']}** ({st.session_state.production_model['version']})")
        
        tab1, tab2 = st.tabs(["✍️ Single Feature Input Quote", "📁 Batch CSV Predictions"])

        X_sample = st.session_state.dataset.drop(columns=[st.session_state.target_col, "travelCode", "userCode", "date"], errors="ignore")

        with tab1:
            st.subheader("Feature Values Input")
            input_data = {}
            cols = st.columns(3)
            
            for i, col_name in enumerate(X_sample.columns):
                with cols[i % 3]:
                    if np.issubdtype(X_sample[col_name].dtype, np.number):
                        default_val = float(X_sample[col_name].median())
                        input_data[col_name] = st.number_input(f"{col_name}", value=default_val)
                    else:
                        unique_vals = X_sample[col_name].dropna().unique().tolist()
                        input_data[col_name] = st.selectbox(f"{col_name}", unique_vals)

            if st.button("🔮 Generate Prediction"):
                input_df = pd.DataFrame([input_data])
                pipeline = st.session_state.production_pipeline
                
                pred_val = pipeline.predict(input_df)[0]
                
                st.markdown("---")
                if st.session_state.problem_type == "Regression":
                    st.metric(
                        f"Predicted {st.session_state.target_col}", 
                        f"${pred_val:,.2f}" if "price" in st.session_state.target_col.lower() else f"{pred_val:.2f}"
                    )
                else:
                    st.metric(f"Predicted {st.session_state.target_col}", f"Class {pred_val}")

                # 🔍 Feature Importance Chart
                if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                    st.markdown("---")
                    st.subheader("🔍 Feature Importance Impact")
                    importances = pipeline.named_steps['classifier'].feature_importances_
                    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    fig_imp = px.bar(
                        x=importances[:len(num_cols)], 
                        y=num_cols, 
                        orientation='h',
                        title="Feature Importances Weight Breakdown",
                        labels={'x': 'Importance Score', 'y': 'Feature'}
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

        with tab2:
            batch_file = st.file_uploader("Upload Batch CSV for Predictions", type=["csv"], key="batch_pred")
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("⚙️ Execute Batch Predictions"):
                    pipeline = st.session_state.production_pipeline
                    batch_df["Prediction"] = pipeline.predict(batch_df)
                    st.success("Batch Inference Completed!")
                    st.dataframe(batch_df, use_container_width=True)

# ==========================================
# MODULE 9 & 10: MONITORING & DATA DRIFT
# ==========================================
elif menu_option == "📈 Performance Monitoring" or menu_option == "🚨 Data Drift Detection":
    st.title("🚨 Dataset Drift & Telemetry Monitoring")
    df_train = st.session_state.dataset

    if df_train is not None:
        st.subheader("Simulate Production Feature Data Drift")
        drift_factor = st.slider("Simulate Drift Scale Multiplier", 1.0, 2.5, 1.0, 0.1)

        df_prod = df_train.copy()
        num_cols = df_prod.select_dtypes(include=[np.number]).columns.tolist()
        if st.session_state.target_col in num_cols:
            num_cols.remove(st.session_state.target_col)

        for c in num_cols:
            df_prod[c] = df_prod[c] * drift_factor

        st.subheader("📊 Statistical Drift Indicators (PSI & KS Test)")
        
        drift_results = []
        for c in num_cols[:5]:
            psi_score = calculate_psi(df_train[c], df_prod[c])
            ks_stat, p_val = ks_2samp(df_train[c].dropna(), df_prod[c].dropna())
            
            status = "Normal 🟢"
            if psi_score >= 0.25:
                status = "Drift Detected 🚨"
            elif psi_score >= 0.10:
                status = "Warning ⚠️"

            drift_results.append({
                "Feature": c,
                "PSI Score": psi_score,
                "KS Test p-value": round(p_val, 4),
                "Drift Status": status
            })

        drift_df = pd.DataFrame(drift_results)
        st.dataframe(drift_df, use_container_width=True)

        # 🔔 Real-Time Drift Alert Simulation & Toast Popups
        if any(d["PSI Score"] >= 0.25 for d in drift_results):
            st.error("🚨 CRITICAL ALERT: Data drift threshold exceeded (PSI > 0.25)!")
            st.toast("Alert: Production Data Drift Detected!", icon="🚨")
