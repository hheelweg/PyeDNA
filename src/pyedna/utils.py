import glob
import os
import sys

# clean *.joblib files 
def cleanCache():
    for file in glob.glob("*.joblib"):
        os.remove(file)


# find unique file with specific extension
def findFileWithExtension(desired_extension):

    cwd = os.getcwd()

    # List all files in the current working directory with the desired extension
    matching_files = [f for f in os.listdir(cwd) if f.endswith(desired_extension)]

    # Check the number of matching files and allow only one
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file with the extension '{desired_extension}' found in {cwd}.")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files with the extension '{desired_extension}' found in {cwd}.")
    else:
        # Exactly one matching file
        target_file = matching_files[0]
        return target_file
    

# find unique file with specific name
def findFileWithName(desired_name):

    cwd = os.getcwd()

    # List all files in the current working directory with the desired name
    matching_files = [f for f in os.listdir(cwd) if f.endswith(desired_name)]

    # Check the number of matching files and allow only one
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file with the name '{desired_name}' found in {cwd}.")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files with the name '{desired_name}' found in {cwd}.")
    else:
        # Exactly one matching file
        target_file = matching_files[0]
        return target_file