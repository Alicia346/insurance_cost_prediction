import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Charger le dataset
df = pd.read_csv("./data/insurance.csv")

# 2. Séparer X (features) et y (target)
X = df.drop("charges", axis=1)
y = df["charges"]

# 3. One-Hot Encoding des variables catégorielles
X = pd.get_dummies(
    X,
    columns=["sex", "smoker", "region"],
    drop_first=True  # on enlève une catégorie de chaque pour éviter les colonnes en trop
)

# 4. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 5. Définir les modèles à comparer
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),
    "KNeighborsRegressor": KNeighborsRegressor(
        n_neighbors=5
    ),
}

results = []
best_model = None
best_model_name = None
best_r2 = -999

# 6. Entraîner et évaluer chaque modèle
for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle : {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"{name} → MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

    # Mettre à jour le meilleur modèle (basé sur R²)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# 7. Afficher un tableau récapitulatif trié par R²
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False)
print("\n📊 Résultats comparés :")
print(results_df)

print(f"\n✅ Meilleur modèle : {best_model_name} avec R² = {best_r2:.4f}")

# 8. Sauvegarder le meilleur modèle
with open("./models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n💾 Modèle sauvegardé dans ./models/best_model.pkl")
