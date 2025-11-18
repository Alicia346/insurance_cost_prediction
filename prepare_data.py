import pandas as pd

# 1. Charger le dataset
df = pd.read_csv("./data/insurance.csv")

# 2. Afficher les colonnes pour vérification
print("Colonnes :", df.columns.tolist())

# 3. Vérifier s'il y a des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# 4. Afficher les premières lignes
print("\nAperçu des données :")
print(df.head())

# 5. Afficher les types de données
print("\nTypes des colonnes :")
print(df.dtypes)
