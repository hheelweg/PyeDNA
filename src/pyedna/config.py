# file to handle global variables
import os

# Retrieve the PyeDNA home directory from the environment variable
PROJECT_HOME = os.getenv('PYEDNA_HOME')

if not PROJECT_HOME:
    raise EnvironmentError("PYEDNA_HOME environment variable is not set.")