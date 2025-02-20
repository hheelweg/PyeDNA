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

    # Check the number of matching files
    if len(matching_files) == 0:
        print(f"No files with the extension '{desired_extension}' found in {cwd}.")
        sys.exit(1)
    elif len(matching_files) > 1:
        print(f"Multiple files with the extension '{desired_extension}' found in {cwd}:")
        sys.exit(1)
    else:
        # Exactly one matching file
        target_file = matching_files[0]
        return target_file