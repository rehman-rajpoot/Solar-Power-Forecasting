# Ye cell "src/load_data.py" naam ki file bana kar save kar dega.

from pathlib import Path
import pandas as pd

FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "sensor1": "Plant_1_Weather_Sensor_Data.csv",
    "gen2": "Plant_2_Generation_Data.csv",
    "sensor2": "Plant_2_Weather_Sensor_Data.csv",
}
# Dictionary: short key → actual CSV filename, sab 4 files (dono plants) ke liye.

def load_raw(data_dir="data"):
    """Return a dict of the four raw DataFrames.

    Keys: 'gen1', 'sensor1', 'gen2', 'sensor2'.
    DATE_TIME is left as a plain string on purpose.
    """
    # Docstring khud keh raha hai: DATE_TIME jaan boojh kar string chhoda gaya hai -
    # is function ka kaam sirf "load karna" hai, "date parse karna" nahi.

    data_dir = Path(data_dir)
    # "data" folder ka path object banaya.

    frames = {}
    for key, name in FILES.items():
        path = data_dir / name
        # e.g. path = data/Plant_1_Generation_Data.csv

        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download the dataset first.")
        # Agar file waha nahi mili, saaf error do - silently fail mat karo.

        frames[key] = pd.read_csv(path)
        # CSV ko as-is load kiya, koi parsing/cleaning nahi ki.

    return frames
    # Sab 4 raw DataFrames wapis kiye.

if __name__ == "__main__":
    # Ye block sirf tab chalega jab file directly run ki jaye:
    # python src/load_data.py  (ya !python src/load_data.py Colab mein)

    raw = load_raw()
    for key, df in raw.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        # Har file ke rows aur columns ka count dikhaya.

        print(df.head(3).to_string(), end="\n\n")
        # Pehli 3 rows print ki taake data ka structure dikh sake -
        # yahi wo jagah hai jaha aap khud DATE_TIME ka raw format dekh sakti hain.
