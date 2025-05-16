import pandas as pd
import sqlite3

# Path to your SQLite database
db_path = "data/mydata.sqlite"  # Replace with your actual database path

# Connect to the database
conn = sqlite3.connect(db_path)

# Run a query and load into DataFrame
query = "SELECT * FROM api_data;"  # Replace with your actual table name
df = pd.read_sql_query(query, conn)
df = pd.read_parquet('backend/db/merged_gurbani.parquet')  # Sample 5 rows for testing

# Step 1: Sort by shabadId AND verseId to maintain original order
df_sorted = df.sort_values(['shabadId', 'verseId'])

# Step 2: Group while preserving ALL verse data
shabad_df = (
    df_sorted.groupby('shabadId')
    .agg({
        'verse': list,           # Ordered list of verses
        'verseId': list,
        'translation_en': list,
        'translation_pu': list,       # Corresponding unique verse IDs
        'ang': 'first',          # Metadata
        'raag': 'first',
        'writer': 'first'
    })
    .reset_index()
)

# Step 3: Create joined text for embeddings
shabad_df['shabad_text'] = shabad_df['verse'].str.join('\n')

# ----------------------------------------------------------------------------------
# VERIFICATION
# ----------------------------------------------------------------------------------

print(f"\n{'='*50}\nVERIFICATION\n{'='*50}")
print(f"Total Shabads: {len(shabad_df)}")
print(f"Total Verses: {sum(len(v) for v in shabad_df['verse'])}")

# Show sample shabad
sample = shabad_df.iloc[0]
print(f"\nSample Shabad ({sample['shabadId']}):")
print(f"Ang {sample['ang']} | Raag {sample['raag']} | {sample['writer']}")
for vid, vtext, trans_en, trans_pu in zip(sample['verseId'], sample['verse'], sample['translation_en'], sample['translation_pu']):
    print(f"  VerseID {vid}: {vtext}")
    print(f"  Translation:{trans_en},\n {trans_pu}")
print(f"columns: {shabad_df.columns.tolist()}")
# Check for single-verse shabads
single_verse = shabad_df[shabad_df['verse'].str.len() == 1]
print(f"\nSingle-verse shabads: {len(single_verse)} (expected in some cases)")

shabad_df.to_parquet('backend/db/merged_shabad.parquet', index=False)
