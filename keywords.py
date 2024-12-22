import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
import requests
from dotenv import load_dotenv
import os

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

nltk.download('stopwords')

# Função para extrair palavras-chave
def extract_keywords(text, num_keywords=5):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop_words = set(stopwords.words('portuguese'))
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(num_keywords)]

# Função para buscar vídeos no YouTube com combinação de palavras-chave
def search_youtube_videos(keywords, max_results=5):
    youtube_api_url = "https://www.googleapis.com/youtube/v3/search"

    if not YOUTUBE_API_KEY:
        raise ValueError("YouTube API key not found. Please set it in the .env file.")

    search_query = " ".join(keywords)

    params = {
        "part": "snippet",
        "q": search_query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
    }

    response = requests.get(youtube_api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        video_info = [
            {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            }
            for item in data.get("items", [])[:max_results]
        ]
        return video_info
    else:
        return f"Error: {response.status_code} - {response.text}"
