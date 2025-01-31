import os
import time
import json
import pickle
import logging
from datetime import datetime
from collections import deque
from typing import Deque, Set, Dict, Any
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Constants
DEFAULT_SETTINGS = {
    "polling_interval": 5,
    "cooldown": 2,
    "response_mode": "all",
    "auto_greet": True,
    "language": "hinglish",
    "jokes": True,
    "vip_users": [],
    "blacklisted_users": [],
}

# Load GPT-Neo model once
logger.info("Loading GPT-Neo model...")
generator = pipeline(
    'text-generation',
    model='EleutherAI/gpt-neo-1.3B',
    device=0 if torch.cuda.is_available() else -1
)
logger.info("Model loaded!")

# Environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
LIVE_CHAT_ID = os.getenv("LIVE_CHAT_ID")

# File paths
TOKEN_FILE = "token.pickle"
SETTINGS_FILE = "settings.json"

def load_settings() -> Dict[str, Any]:
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading settings: {e}. Using defaults.")
        return DEFAULT_SETTINGS

def save_settings(settings: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Error saving settings: {e}")

def retry_api_call(func, *args, **kwargs):
    retries = 5
    delay = 1
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except HttpError as e:
            logger.error(f"API error: {e}")
            if e.resp.status in [500, 503]:
                time.sleep(delay)
                delay *= 2
            elif e.resp.status == 403:
                logger.error("Quota exceeded. Stopping retries.")
                return None
            else:
                break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def authenticate_youtube():
    creds = None
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secret.json',
                    ['https://www.googleapis.com/auth/youtube.force-ssl']
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)
        return build('youtube', 'v3', credentials=creds)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

def generate_response(prompt: str) -> str:
    try:
        response = generator(
            prompt,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return response[0]['generated_text'].strip()
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "Sorry, couldn't process your request. Please try again!"

def handle_super_chat(youtube, live_chat_id: str, message: Dict, author_name: str) -> None:
    try:
        amount = message['snippet']['superChatDetails']['amountDisplayString']
        response = f"Thanks {author_name} for the {amount} Super Chat! 🙌"
    except KeyError:
        response = f"Thanks {author_name} for the support! 💖"
    
    retry_api_call(
        youtube.liveChatMessages().insert,
        part='snippet',
        body={
            'snippet': {
                'liveChatId': live_chat_id,
                'type': 'textMessageEvent',
                'textMessageDetails': {
                    'messageText': response[:500]  # Enforce character limit
                }
            }
        }
    )

def main():
    settings = load_settings()
    youtube = authenticate_youtube()
    
    last_seen_messages = set()
    new_users = set()
    priority_queue = deque()
    normal_queue = deque()

    try:
        while True:
            # Fetch and process messages
            response = retry_api_call(
                youtube.liveChatMessages().list,
                liveChatId=LIVE_CHAT_ID,
                part='snippet,authorDetails',
                maxResults=50
            )

            if response:
                for msg in response.get('items', []):
                    process_message(msg, last_seen_messages, new_users, priority_queue, normal_queue)

            # Process queues
            process_queues(youtube, priority_queue, normal_queue)
            
            # Dynamic sleep based on settings
            time.sleep(max(1, settings.get('polling_interval', 5)))

    except Exception as e:
        logger.error(f"Critical failure: {e}")
    finally:
        logger.info("Bot shutting down. Saving state...")
        save_settings(settings)

if __name__ == "__main__":
    main()