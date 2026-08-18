import pandas as pd
import datetime

def run():
    print(f"[{datetime.datetime.now()}] Custom Python container started successfully!")
    df = pd.DataFrame({"Task": ["Extract", "Transform", "Load"], "Status": ["OK", "OK", "OK"]})
    print(df)
    print("Execution complete.")

if __name__ == "__main__":
    run()