import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your input CSV
INPUT_CSV = "/home/lukovic/data/treenet/sites.csv"

# Output paths
TRAIN_OUT = "/home/lukovic/data/treenet/train_sites.csv"
TEST_OUT = "/home/lukovic/data/treenet/test_sites.csv"

# Load the CSV
df = pd.read_csv(INPUT_CSV)

# Filter only country == "Switzerland"
df_ch = df[df["country"] == "Switzerland"].copy()

# Random 80/20 split (stratification optional if countries differ)
train_df, test_df = train_test_split(
    df_ch,
    test_size=0.2,
    random_state=42,  # for reproducibility
    shuffle=True
)

# Save only the site_id column
train_df[["site_id"]].to_csv(TRAIN_OUT, index=False)
test_df[["site_id"]].to_csv(TEST_OUT, index=False)

print(f"Train sites saved to: {TRAIN_OUT} ({len(train_df)} rows)")
print(f"Test sites saved to:  {TEST_OUT} ({len(test_df)} rows)")
