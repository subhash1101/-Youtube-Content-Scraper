from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import os
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import whisper
import re
from transformers import pipeline
from deep_translator import GoogleTranslator
import logging
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from crawl4ai import WebCrawler
import json
import boto3
from pydantic import BaseModel


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# YouTube API key (use environment variables for security)
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyDUiH26pHYFrUgF4xLOSNjfgr2-aziltYk')

# Initialize Whisper model globally
whisper_model = whisper.load_model("base")

# Initialize summarizer globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Set up logging
logging.basicConfig(level=logging.INFO)

# YouTube API details
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# S3 setup
s3_client = boto3.client('s3')
S3_BUCKET_NAME = 'ytscriber'  # Replace with your S3 bucket name
crawler = WebCrawler()
crawler.warmup()

def get_channel_id_from_url(channel_url):
    match = re.search(r'(channel/|user/|@)([A-Za-z0-9_\-]+)', channel_url)
    return match.group(2) if match else None

def get_channel_id(youtube, handle):
    response = youtube.channels().list(
        part="id",
        forUsername=handle
    ).execute()
    if "items" in response and response["items"]:
        return response["items"][0]["id"]
    
    response = youtube.search().list(
        part="snippet",
        q=handle,
        type="channel",
        maxResults=1
    ).execute()
    if "items" in response and response["items"]:
        return response["items"][0]["snippet"]["channelId"]
    
    return None

def fetch_latest_videos(channel_url, max_results=5):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    handle = get_channel_id_from_url(channel_url)
    channel_id = get_channel_id(youtube, handle)
    
    if not channel_id:
        logging.error("Invalid channel URL or handle!")
        return []

    search_response = youtube.search().list(
        channelId=channel_id,
        part='snippet',
        order='date',
        maxResults=max_results
    ).execute()
    
    videos = [
        {
            'title': item['snippet']['title'],
            'videoId': item['id']['videoId'],
            'thumbnail': item['snippet']['thumbnails']['high']['url']
        }
        for item in search_response.get('items', [])
        if 'videoId' in item['id']
    ]
    
    return videos

def download_youtube_audio(url, filename="youtube_audio"):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{filename}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return f"{filename}.mp3"
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def convert_mp3_to_wav(mp3_file, wav_file="audio.wav"):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        logging.error(f"Error converting MP3 to WAV: {e}")
        return None

def preprocess_audio(wav_file):
    try:
        audio = AudioSegment.from_wav(wav_file)
        audio = audio + 5  # Increase volume
        if audio.channels > 1:
            audio = audio.set_channels(1)
        processed_wav = f"processed_{wav_file}"
        audio.export(processed_wav, format="wav")
        return processed_wav
    except Exception as e:
        logging.error(f"Error preprocessing audio: {e}")
        return None

def transcribe_audio_to_text_with_timestamps(audio_file, target_language='en'):
    result = whisper_model.transcribe(audio_file)
    segments = result['segments']

    if target_language != 'en':
        segments = translate_transcription_with_timestamps(segments, target_language)

    transcription_with_timestamps = ""
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        transcript_text = segment['text']
        transcription_with_timestamps += f"<div><strong>{format_timestamp(start_time)} - {format_timestamp(end_time)}:</strong> {transcript_text}</div>\n"

    return transcription_with_timestamps

def format_timestamp(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

def translate_transcription_with_timestamps(segments, target_language='es'):
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_segments = []

    for segment in segments:
        try:
            translated_text = translator.translate(segment['text'])
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': translated_text
            })
        except Exception as e:
            logging.error(f"Translation error: {e}")
            translated_segments.append(segment)

    return translated_segments

def chunk_text_by_length(text, max_chunk_size=500):
    sentences = text.split('. ')
    chunks, current_chunk, current_length = [], "", 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk, current_length = sentence + '. ', sentence_length
        else:
            current_chunk += sentence + '. '
            current_length += sentence_length

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def summarize_transcription(transcription_text):
    transcription_text = re.sub('<[^<]+?>', '', transcription_text)
    chunks = chunk_text_by_length(transcription_text, max_chunk_size=500)
    summaries = []

    for i, chunk in enumerate(chunks):
        if chunk.strip():
            try:
                summary = summarizer(chunk, min_length=30, max_length=150, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logging.error(f"Summarization error on chunk {i}: {e}")
                continue

    return ' '.join(summaries)

def save_transcription_and_summary(video_id, transcription, summary):
    file_name = f"{video_id}.txt"
    try:
        with open(file_name, 'w') as f:
            f.write("Transcription:\n")
            f.write(transcription + "\n\n")
            f.write("Summary:\n")
            f.write(summary + "\n")
        logging.info(f"Saved transcription and summary to {file_name}")
    except Exception as e:
        logging.error(f"Error saving transcription and summary: {e}")

def upload_file_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"Uploaded {file_name} to bucket {bucket}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to S3: {e}")
        return False
    
def find_subpages(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    subpages = set()

    for link in links:
        href = link['href']
        full_url = urljoin(base_url, href)
        if full_url.startswith(base_url):
            subpages.add(full_url)
    return subpages

def get_main_word_from_url(base_url):
    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc
    main_word = domain.split('.')[0]
    return main_word

@app.post('/upload_to_s3')
async def upload_to_s3(video_id: str = Form(...), transcription: str = Form(...), summary: str = Form(...)):
    transcription_file = f"{video_id}_transcription.txt"
    summary_file = f"{video_id}_summary.txt"
    
    try:
        with open(transcription_file, 'w') as f:
            f.write(transcription)
        with open(summary_file, 'w') as f:
            f.write(summary)

        transcription_upload = upload_file_to_s3(transcription_file, S3_BUCKET_NAME, f"{video_id}/{transcription_file}")
        summary_upload = upload_file_to_s3(summary_file, S3_BUCKET_NAME, f"{video_id}/{summary_file}")

        if transcription_upload and summary_upload:
            return JSONResponse(content={'status': 'success'})
        else:
            raise HTTPException(status_code=500, detail="Failed to upload files to S3")
    finally:
        if os.path.exists(transcription_file):
            os.remove(transcription_file)
        if os.path.exists(summary_file):
            os.remove(summary_file)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_videos", response_class=HTMLResponse)
async def get_videos(request: Request, channel_url: str = Form(...)):
    if not channel_url:
        return templates.TemplateResponse("error.html", {"request": request, "error": "YouTube channel URL is required"})
    
    # Remove await if fetch_latest_videos is not asynchronous
    videos = await fetch_latest_videos(channel_url) if callable(fetch_latest_videos) and fetch_latest_videos.__name__ == "coroutine" else fetch_latest_videos(channel_url)
    return templates.TemplateResponse("videos_overview.html", {"request": request, "videos": videos, "channel_url": channel_url})

@app.post('/delete_video')
async def delete_video(request: Request):
    data = await request.json()  # Extract JSON data from request
    video_id = data.get('video_id')  # Safely access the video_id key

    if not video_id:
        return JSONResponse(content={'success': False, 'error': 'Video ID not provided'}, status_code=400)

    # Add any additional logic for deleting the video here

    return JSONResponse(content={'success': True})

@app.post("/transcribe_remaining", response_class=HTMLResponse)
async def transcribe_remaining(request: Request):
    data = await request.json()
    remaining_video_ids = data.get("video_ids", [])
    channel_id = data.get("channel_id")

    if not remaining_video_ids:
        raise HTTPException(status_code=400, detail="No videos to transcribe")

    videos = []
    for video_id in remaining_video_ids:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        audio_file = download_youtube_audio(youtube_url)

        if not audio_file:
            continue

        wav_file = convert_mp3_to_wav(audio_file)
        if not wav_file:
            continue

        processed_wav = preprocess_audio(wav_file)
        transcription = transcribe_audio_to_text_with_timestamps(processed_wav)
        summary = summarize_transcription(transcription)

        # Save transcription and summary to a file
        save_transcription_and_summary(video_id, transcription, summary)

        videos.append({
            'videoId': video_id,
            'transcription': transcription,
            'summary': summary,
            'title': f'Video {video_id}'
        })

    # Render the 'videos.html' template with the transcribed videos
    return templates.TemplateResponse("videos.html", {"request": request, "videos": videos})

@app.post("/crawl_website")
async def crawl_website(url: str = Form(...)):
    subpages = find_subpages(url)
    main_word = get_main_word_from_url(url)
    data = []

    with open('crawler_results.txt', 'w', encoding='utf-8') as f_text:
        for i in subpages:
            result = crawler.run(url=i)
            if result.markdown != "None":
                f_text.write(f"URL: {i}\nContent:\n{result.markdown}\n" + "="*50 + "\n")
                data.append({"url": i, "content": result.markdown})

    s3_client.upload_file('crawler_results.txt', S3_BUCKET_NAME, f"{main_word}.txt")

    with open('crawler_results.json', 'w', encoding='utf-8') as f_json:
        json.dump(data, f_json, indent=4)

    return JSONResponse({'status': 'success', 'message': 'Crawling completed and data saved to S3'})

@app.get("/video_details/{video_id}", response_class=HTMLResponse)
async def video_details(request: Request, video_id: str):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Ensure each function runs correctly and returns expected results
    audio_file = download_youtube_audio(youtube_url)
    if not audio_file:
        return templates.TemplateResponse("error.html", {"request": request, "error": "Error downloading video audio"})

    wav_file = convert_mp3_to_wav(audio_file)
    if not wav_file:
        return templates.TemplateResponse("error.html", {"request": request, "error": "Error converting to WAV"})

    processed_wav = preprocess_audio(wav_file)
    transcription = transcribe_audio_to_text_with_timestamps(processed_wav)
    summary = summarize_transcription(transcription)

    # Prepare video data to send to the template
    video = {
        'videoId': video_id,
        'transcription': transcription,
        'summary': summary,
        'title': f'Video {video_id}'
    }

    # Render videos.html template with the video data
    return templates.TemplateResponse("videos.html", {"request": request, "videos": [video]})

class TranslateRequest(BaseModel):
    target_language: str = "es"

@app.post("/translate/{video_id}")
async def translate_video(video_id: str, request: TranslateRequest):
    target_language = request.target_language
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    # Step 1: Download the audio from the YouTube video
    audio_file = download_youtube_audio(youtube_url)
    if not audio_file:
        raise HTTPException(status_code=500, detail="Failed to download audio")

    # Step 2: Convert MP3 to WAV format
    wav_file = convert_mp3_to_wav(audio_file)
    if not wav_file:
        raise HTTPException(status_code=500, detail="Failed to process audio")

    # Step 3: Preprocess audio and transcribe
    processed_wav = preprocess_audio(wav_file)
    transcription = transcribe_audio_to_text_with_timestamps(processed_wav)

    # Step 4: Translate transcription to target language
    translation = translate_transcription_text(transcription, target_language)
    
    # Save transcription and translation (as summary) if translation was successful
    if translation:
        save_transcription_and_summary(video_id, transcription, translation)
        return JSONResponse({"transcription": translation})
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to translate transcription for language: {target_language}. Please verify language code."
        )

def translate_transcription_text(transcription, target_language):
    try:
        logging.info(f"Starting translation to {target_language}")
        translator = GoogleTranslator(source='auto', target=target_language)
        translation = translator.translate(transcription)
        logging.info(f"Translation successful: {translation[:100]}...")  # Log first 100 chars
        return translation
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return None

