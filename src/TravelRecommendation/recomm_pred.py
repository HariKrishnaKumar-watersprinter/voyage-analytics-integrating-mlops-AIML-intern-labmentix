import os
import sys
import pathlib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
from src.TravelRecommendation.travelrecom import travel_rec
def get_latest_model_uri(model_name: str = "cosine_similarity_model") -> str | None:
    client = MlflowClient()
    try:
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            raise ValueError(f"No registered model found with name '{model_name}'")

        latest_version = max(model_versions, key=lambda mv: int(mv.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"
        print(f"Latest version found: v{latest_version.version}")
        return model_uri
    except Exception as e:
        print(f"Error getting latest model: {e}")
        return None


def load_model_for_inference(model_name: str = "cosine_similarity_model"):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("Environment variable MLFLOW_TRACKING_URI is not set")

    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using MLflow Tracking URI: {tracking_uri}")

    model_uri = get_latest_model_uri(model_name)
    if not model_uri:
        return None

    print(f"Loading model from URI: {model_uri}")
    try:
        #mlflow.pyfunc.get_model_dependencies(model_uri)
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_underlying_model(pyfunc_model):
    """
    Safely extract the real Python model object from a pyfunc wrapper.
    """
    # Method 1 (recommended in recent MLflow)
    try:
        return pyfunc_model.unwrap_python_model()
    except Exception:
        pass

    # Method 2
    try:
        return pyfunc_model._model_impl.python_model
    except Exception:
        pass

    # Method 3 (last resort)
    try:
        return pyfunc_model._model_impl
    except Exception:
        pass

    raise AttributeError("Could not extract the underlying Python model")


def stramlit_trav_recomm():
    # Retrieve the unwrapped model directly from session state
    if "unwrapped_rec_model" not in st.session_state:
        st.error("Model is not loaded. Please restart the application.")
        return
        
    rec = st.session_state.unwrapped_rec_model
    hotel = travel_rec()

    options = {
        "places": list(hotel["place"].unique()),
        "hotel_price_per_day_range": (0, float(hotel["price"].max())),
    }
    
    st.title("🏨 Travel Recommendations")
    c1, c2 = st.columns(2)
    place = c1.selectbox("Destination", ["(any)"] + options["places"])
    lo, hi = options["hotel_price_per_day_range"]
    budget = c2.slider("Max budget / night (R$)", float(lo), float(hi), float(hi), 10.0)
    topn = st.slider("Show top N", 3, 10, 5)
    uc = st.selectbox("Personalise for user", ["(anonymous)"] +
                      [str(u) for u in list(rec.user_places.index)[:80]])
    def _cast(u):
        try: return int(u)
        except (TypeError, ValueError): return u

    df = rec.recommend(userCode=None if uc.startswith("(") else _cast(uc),
                       place=None if place.startswith("(") else place,
                       budget_per_day=budget, top_n=topn)
    if df.empty:
        st.warning("No hotels match those filters.")
    else:
        st.dataframe(df[["name", "place", "bookings", "avg_days",
                         "avg_price_per_day", "avg_total", "score", "visited_before"]],
                     width='content')
        st.bar_chart(df.set_index("name")["score"])


# ---- Streamlit App Entry Point ----
if __name__ == "__main__":
    # 1. Initialize the model in session state ONLY ONCE
    if "unwrapped_rec_model" not in st.session_state:
        with st.spinner("Loading MLflow model... This might take a minute on the first run."):
            pyfunc_model = load_model_for_inference(model_name="cosine_similarity_model")
            
            if pyfunc_model is None:
                st.error("Failed to load model. Exiting.")
                st.stop() # Stops the Streamlit app execution elegantly
                
            try:
                # Unwrap and save to session state
                st.session_state.unwrapped_rec_model = get_underlying_model(pyfunc_model)
                st.success("Model loaded and cached successfully!")
            except Exception as e:
                st.error(f"Failed to unwrap model: {e}")
                st.stop()

    # 2. Run the Streamlit UI (uses the cached model on subsequent slider changes)
    stramlit_trav_recomm()