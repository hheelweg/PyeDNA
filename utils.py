import glob
import os

def cleanCache():
    for file in glob.glob("*.joblib"):
        os.remove(file)