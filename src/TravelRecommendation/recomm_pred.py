import os
import sys
import pathlib
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))


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


def recomm_pred(pyfunc_model):
    if pyfunc_model is None:
        print("No model available for recommendation.")
        return

    # Get the real TravelRecommender instance
    try:
        rec = get_underlying_model(pyfunc_model)
        print(f"Successfully unwrapped model. Type: {type(rec)}")
    except Exception as e:
        print(f"Failed to unwrap model: {e}")
        return

    # ---- Known user recommendation ----
    try:
        u = rec.user_places.index[0]
        print("\nTop-10 for known user:")
        result = rec.recommend(userCode=u, top_n=10)
        print(result[["name", "place", "avg_price_per_day", "score"]])
    except Exception as e:
        print(f"Could not generate recommendations for known user: {e}")

    # ---- Interactive recommendation ----
    try:
        rec_place = input("\nEnter the place you want to visit: ").strip()
        rec_bud = float(input("Please enter your budget per day: "))

        print(f"\nTop-5 in {rec_place} under R${rec_bud}/night:")
        rec_result = rec.recommend(
            place=rec_place,
            budget_per_day=rec_bud,
            top_n=5
        )

        if rec_result is not None and not rec_result.empty:
            print(rec_result[["name", "avg_price_per_day", "popularity", "score"]])
        else:
            print("No recommendations found.")
    except Exception as e:
        print(f"Error during interactive recommendation: {e}")


if __name__ == "__main__":
    model = load_model_for_inference(model_name="cosine_similarity_model")

    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    # ---- Option A: Use .predict() (with correct dtypes) ----
    try:
        input_data = pd.DataFrame({
            "userCode": [1.0],                          # float
            "place": ["Florianopolis (SC)"],
            "budget_per_day": [100.0],
            #"top_n": [5.0]                              # float (not int)
        })
        predictions = model.predict(input_data)
        print("\nPredictions via .predict():")
        print(predictions)
    except Exception as e:
        print(f"\n.predict() failed: {e}")

    # ---- Option B: Use custom .recommend() method ----
    recomm_pred(model)