# Plant 1 aur Plant 2 ke raw generation aur sensor datasets load karne ke liye helper module

from pathlib import Path
import pandas as pd

FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "sensor1": "Plant_1_Weather_Sensor_Data.csv",
    "gen2": "Plant_2_Generation_Data.csv",
    "sensor2": "Plant_2_Weather_Sensor_Data.csv",
}
# Short keys map kiye actual CSV filenames ke sath (dono plants ke liye)

def load_raw(data_dir="data"):
    """Return a dict of the four raw DataFrames.

    Keys: 'gen1', 'sensor1', 'gen2', 'sensor2'.
    DATE_TIME is left as a plain string on purpose.
    """
    # DATE_TIME ko string ke tor par load kia hai - datetime parsing prepare.py mein alag se ki hai

    data_dir = Path(data_dir)
    # Data directory path object banaya

    frames = {}
    for key, name in FILES.items():
        path = data_dir / name

        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download the dataset first.")
        # Agar file exist nahi karti tou clear FileNotFoundError raise kia

        frames[key] = pd.read_csv(path)
        # Raw CSV file load ki without pre-cleaning

    return frames
    # Chaaro raw DataFrames dictionary format mein return kiye

if __name__ == "__main__":
    raw = load_raw()
    for key, df in raw.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        # Har file ke row aur column dimensions print kiye

        print(df.head(3).to_string(), end="\n\n")
        # Initial structure aur raw datetime format verify karne ke liye pehli 3 rows inspect kiye
