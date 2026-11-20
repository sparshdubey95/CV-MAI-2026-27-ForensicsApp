"""Convenient entry point for students: ``python run.py``."""

import os

# Suppress TensorFlow oneDNN and deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from forensics_app.app import main


if __name__ == "__main__":
    main()
