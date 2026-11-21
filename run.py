"""Convenient entry point for students: ``python run.py``."""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from forensics_app.app import main


if __name__ == "__main__":
    main()
