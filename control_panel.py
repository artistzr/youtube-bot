import streamlit as st
import json
import re

EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

def validate_username(username: str) -> bool:
    return 3 <= len(username) <= 25 and username.isalnum()

# ... (rest of the control panel code from previous version)
# Add input validation for all user inputs