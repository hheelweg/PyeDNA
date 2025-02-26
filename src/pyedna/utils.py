import glob
import os
import sys

# clean *.joblib files 
def cleanCache():
    for file in glob.glob("*.joblib"):
        os.remove(file)


# find unique file with specific extension (dir = None means cwd)
def findFileWithExtension(desired_extension, dir = None):

    search_dir = dir if dir else os.getcwd()

    # List all files in the current working directory with the desired extension
    matching_files = [f for f in os.listdir(search_dir) if f.endswith(desired_extension)]

    # Check the number of matching files and allow only one
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file with the extension '{desired_extension}' found in {search_dir}.")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files with the extension '{desired_extension}' found in {search_dir}.")
    else:
        # Exactly one matching file
        target_file = matching_files[0]
        return os.path.join(search_dir, target_file)
    

# find unique file with specific name (dir = None means cwd)
def findFileWithName(desired_name, dir = None):

    search_dir = dir if dir else os.getcwd()

    # List all files in the current working directory with the desired name
    matching_files = [f for f in os.listdir(search_dir) if f.endswith(desired_name)]

    # Check the number of matching files and allow only one
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file with the name '{desired_name}' found in {search_dir}.")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files with the name '{desired_name}' found in {search_dir}.")
    else:
        # Exactly one matching file
        target_file = matching_files[0]
        return os.path.join(search_dir, target_file)

# do check whether file with specified name exists in dir (dir = None means cwd)
def checkFileWithName(desired_name, dir = None):
    search_dir = dir if dir else os.getcwd()

    # List all files in the current working directory with the desired name
    matching_files = [f for f in os.listdir(search_dir) if f.endswith(desired_name)]

    # Check the number of matching files and allow only one
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file with the name '{desired_name}' found in {search_dir}.")
    else:
        pass

# in some specified directory, find directories with matching name
def findSubdirWithName(desired_name, dir):

    if not os.path.isdir(dir):
        raise Warning(f"Base directory '{dir}' does not exist.")

    target_dir = os.path.join(dir, desired_name)
    if os.path.isdir(target_dir):
        return target_dir
    else:
        raise NameError(f"No matching directory for {desired_name} found in {dir}")