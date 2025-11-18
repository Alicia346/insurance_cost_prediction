import pandas as pd

# Charger le dataset
df = pd.read_csv("./data/insurance.csv")

# Voir les premières lignes
print(df.head())

# Informations générales
print(df.info())
