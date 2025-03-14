import pandas as pd

# Läs in CSV-filen
df = pd.read_csv('../data/customer_data.csv', delimiter=';')

# Gruppera data efter 'Primary key' och 'InternalName' och aggregera värdena
df_agg = df.groupby(['Primary key', 'InternalName']).agg({
    'OptOut': 'max',
    'Open': 'max',
    'Click': 'max',
    'Gender': 'first',
    'Age': 'first',
    'Bolag': 'first'
}).reset_index()

# Spara den aggregerade data till en ny CSV-fil
df_agg.to_csv('../data/new_customer_data.csv', index=False, sep=';')

print("Fil omvandlad och sparad som 'new_customer_data.csv'")