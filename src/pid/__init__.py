import os

from .env import load_dotenv

if not os.getenv("PID_NO_DOTENV"):
    load_dotenv()  # reads ./.env if present, so `python -m pid.smoke llm` works without exporting variables by hand
