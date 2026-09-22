import pandas as pd
import numpy as np
import sys
import pathlib
import os
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PythonModel # Import PythonModel base class
from mlflow.tracking import MlflowClient
import dagshub
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from database import engine


load_dotenv()
def travel_rec():
    hotel = pd.read_sql("SELECT * FROM hotel", engine)
    hotel["date"]  = pd.to_datetime(hotel["date"], errors="coerce")
    
    # Fix: reassign the dataframe after dropping duplicates and nulls
    hotel = hotel.drop_duplicates().dropna()
    
    recalc = ~np.isclose(hotel["total"],
                        hotel["days"] * hotel["price"], rtol=0.05)
    hotel.loc[recalc, "total"] = (hotel.loc[recalc, "days"]
                                         * hotel.loc[recalc, "price"])
    
    hotel["month"] = hotel["date"].dt.month
    hotel["quarter"] = hotel["date"].dt.quarter
    hotel["day_of_week"] = hotel["date"].dt.dayofweek
    hotel["is_weekend"] = (hotel["day_of_week"] >= 5).astype(int)    
    
    # Note: hprof is calculated but never returned/used. Leaving as is.
    hprof = (hotel.groupby("userCode")
             .agg(total_stays=("travelCode", "count"), avg_hotel_days=("days", "mean"),
                  total_hotel_spend=("total", "sum"),
                  avg_hotel_price=("price", "mean"),
                  n_hotels=("name", "nunique")).reset_index())
    return hotel

# Inherit from mlflow.pyfunc.PythonModel
class TravelRecommender(PythonModel):
    def fit(self, hotel: pd.DataFrame):
        self.hotel = hotel.copy()
        self.hotel_stats = (hotel.groupby(["name", "place"], as_index=False)
                            .agg(bookings=("travelCode", "nunique"),
                                 avg_days=("days", "mean"),
                                 avg_price_per_day=("price", "mean"),
                                 avg_total=("total", "mean")))
        self.hotel_stats["popularity"] = (self.hotel_stats["bookings"]
                                          / self.hotel_stats["bookings"].max())
        self.place_matrix = pd.crosstab(hotel["userCode"], hotel["place"])
        
        sim = cosine_similarity(self.place_matrix)
        self.sim = pd.DataFrame(sim, index=self.place_matrix.index,
                                columns=self.place_matrix.index)
        self.user_places = (hotel.groupby("userCode")["place"]
                            .agg(lambda s: sorted(set(s))))
        return self

    def _cf_scores(self, userCode):
        if userCode not in self.sim.index:
            return pd.Series(dtype=float)
        sims = self.sim.loc[userCode].drop(index=userCode).sort_values(ascending=False).head(10)
        nb = (self.hotel[self.hotel["userCode"].isin(sims.index)]
              .groupby("name")["travelCode"].nunique())
        w = sims.mean()
        return nb / (w if w else 1.0)

    def recommend(self, userCode=None, place=None, budget_per_day=None, top_n=5):
        cand = self.hotel_stats.copy()
        
        # 1. Filter by Place
        if place:
            cand = cand[cand["place"].str.lower() == str(place).lower()]
            
        # 2. Diagnose if empty AFTER place filter
        if cand.empty and place:
            place_exists = self.hotel_stats["place"].str.lower().eq(str(place).lower()).any()
            if not place_exists:
                reason = f"Location '{place}' not found in database."
            else:
                reason = f"Hotels in '{place}' exist, but none match the budget criteria."
            print(f"⚠️ Recommendation Failed: {reason}")
            return None

        # 3. Filter by Budget
        if budget_per_day is not None:
            budget_per_day = float(budget_per_day)
            cand = cand[cand["avg_price_per_day"] <= budget_per_day * 1.15]   # 15% flex
            
        # 4. Diagnose if empty AFTER budget filter
        if cand.empty:
            if budget_per_day is not None:
                min_price = self.hotel_stats[self.hotel_stats["place"].str.lower() == str(place).lower()]["avg_price_per_day"].min()
                reason = (f"Budget too low. The cheapest hotel in '{place}' costs R${min_price:.2f}/night. "
                          f"Your max with 15% flex is R${budget_per_day * 1.15:.2f}/night.")
            else:
                reason = "No hotels available for the given criteria."
            
          
            return print(f"⚠️ Recommendation Failed: {reason}")
        
        # 5. Normal Scoring Logic
        cf = self._cf_scores(userCode) if userCode is not None else pd.Series(dtype=float)
        cand["cf_score"] = cand["name"].map(cf).fillna(0.0)
        
        mx = cand["cf_score"].max()
        if mx > 0:
            cand["score"] = 0.55 * cand["popularity"] + 0.45 * (cand["cf_score"] / mx)
        else:
            cand["score"] = cand["popularity"]
            
        visited = set(self.user_places.get(userCode, [])) if userCode is not None else set()
        cand["visited_before"] = cand["place"].map(lambda p: p in visited)
        
        return cand.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    # Implement the required predict method for MLflow pyfunc
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        model_input is expected to be a DataFrame with columns: 
        'userCode', 'place', 'budget_per_day', 'top_n'
        """
        results = []
        for _, row in model_input.iterrows():
            recs = self.recommend(
                userCode=row.get("userCode"),
                place=row.get("place"),
                budget_per_day=row.get("budget_per_day"),
                top_n=row.get("top_n", 5)
            )
            results.append(recs)
        return pd.DataFrame({"recommendations": results})
       
def recomm():
    hotel = travel_rec()
    rec = TravelRecommender().fit(hotel)
    
    return rec

def log_model_to_mlflow():
    
    
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Voyage Analytics")
    
    MODEL_NAME = "cosine_similarity_model"

    with mlflow.start_run(run_name="cosine_sim_v1") as run:
        mlflow.log_param("similarity", "cosine")
        rec = recomm()
        
        # 1. Create a sample input so MLflow can infer the schema (signature)
        input_example = pd.DataFrame({
            "userCode": pd.Series([1], dtype="float64"),
            "place": ["Florianopolis (SC)"],
            "budget_per_day": [100.0],
            "top_n": pd.Series([5], dtype="float64")
        })
        
        # 2. Infer the signature from the input example and the predict method's type hints
        signature = mlflow.models.infer_signature(input_example, rec.predict(None, input_example))

        mlflow.pyfunc.log_model(
            name="cosine_similarity_model", 
            python_model=rec,
            signature=signature,          # Provides the schema to silence the warning
            input_example=input_example,
            registered_model_name="cosine_similarity_model",  # Provides an example input for the UI
            pip_requirements=["scikit-learn", "pandas", "numpy", "joblib"]
        )

        
    print('Model loaded successfully') 
    return rec


if __name__ == "__main__":
    log_model_to_mlflow()
