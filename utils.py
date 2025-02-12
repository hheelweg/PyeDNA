import glob
import os

def cleanCache():
    for file in glob.glob("*.joblib") + glob.glob("*.npz"):
        os.remove(file)