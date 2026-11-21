"""Allow the application to be started with ``python -m forensics_app``."""

import os
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

from .app import main


if __name__ == "__main__":
    main()
