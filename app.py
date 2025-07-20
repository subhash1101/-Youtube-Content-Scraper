from flask import Flask, render_template, request, jsonify
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
import logging

app = Flask(__name__)

# YouTube API key (use environment variables for security)
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyDUiH26pHYFrUgF4xLOSNjfgr2-aziltYk')

# Initialize Whisper model globally
whisper_model = whisper.load_model("base")

# Initialize summarizer globally
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Set up logging
logging.basicConfig(level=logging.INFO)

# YouTube API key (use environment variables for security)
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


# Get channel ID from YouTube channel URL
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

# YouTube API: Get latest 10 videos from a channel
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

# Download YouTube video as audio
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

# Convert MP3 to WAV format
def convert_mp3_to_wav(mp3_file, wav_file="audio.wav"):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        logging.error(f"Error converting MP3 to WAV: {e}")
        return None

# Preprocess audio: Increase volume and convert to mono
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

# Transcribe audio with timestamps
def transcribe_audio_to_text_with_timestamps(audio_file, target_language='en'):
    result = whisper_model.transcribe(audio_file)
    segments = result['segments']

    # Optionally, translate segments if target_language != 'en'
    if target_language != 'en':
        segments = translate_transcription_with_timestamps(segments, target_language)

    transcription_with_timestamps = ""
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        transcript_text = segment['text']
        transcription_with_timestamps += f"<div><strong>{format_timestamp(start_time)} - {format_timestamp(end_time)}:</strong> {transcript_text}</div>\n"

    return transcription_with_timestamps

# Helper function for timestamp formatting
def format_timestamp(seconds):
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

# Translate transcription segments
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

# Chunk text by length with error handling
def chunk_text_by_length(text, max_chunk_size=500):
    sentences = text.split('. ')
    chunks, current_chunk, current_length = [], "", 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        # If adding this sentence exceeds the chunk size, start a new chunk
        if current_length + sentence_length > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk, current_length = sentence + '. ', sentence_length
        else:
            current_chunk += sentence + '. '
            current_length += sentence_length

    # Append any remaining text as a chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# Enhanced summarization function with error handling for large texts
def summarize_transcription(transcription_text):
    # Strip HTML tags
    transcription_text = re.sub('<[^<]+?>', '', transcription_text)
    chunks = chunk_text_by_length(transcription_text, max_chunk_size=500)
    summaries = []

    for i, chunk in enumerate(chunks):
        if chunk.strip():
            try:
                # Use the summarizer with adjusted min and max lengths for large texts
                summary = summarizer(chunk, min_length=30, max_length=150, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logging.error(f"Summarization error on chunk {i}: {e}")
                # Skip problematic chunks and continue
                continue

    return ' '.join(summaries)

# Save transcription and summary to a text file
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

# Initialize S3 client
s3_client = boto3.client('s3')
S3_BUCKET_NAME = 'ytscriber'  # Replace with your S3 bucket name

def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    if object_name is None:
        object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"Uploaded {file_name} to bucket {bucket}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to S3: {e}")
        return False

@app.route('/upload_to_s3', methods=['POST'])
def upload_to_s3():
    video_id = request.form.get('video_id')
    transcription = request.form.get('transcription')
    summary = request.form.get('summary')
    
    # Save transcription and summary to temporary text files
    transcription_file = f"{video_id}_transcription.txt"
    summary_file = f"{video_id}_summary.txt"
    
    try:
        with open(transcription_file, 'w') as f:
            f.write(transcription)
        with open(summary_file, 'w') as f:
            f.write(summary)

        # Upload both files to S3
        transcription_upload = upload_file_to_s3(transcription_file, S3_BUCKET_NAME, f"{video_id}/{transcription_file}")
        summary_upload = upload_file_to_s3(summary_file, S3_BUCKET_NAME, f"{video_id}/{summary_file}")

        if transcription_upload and summary_upload:
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'failure', 'error': 'Failed to upload files to S3'}), 500

    except Exception as e:
        logging.error(f"Error in /upload_to_s3 route: {e}")
        return jsonify({'status': 'failure', 'error': str(e)}), 500
    finally:
        # Clean up temporary files if needed
        if os.path.exists(transcription_file):
            os.remove(transcription_file)
        if os.path.exists(summary_file):
            os.remove(summary_file)

# Initialize S3 and crawler
s3 = boto3.resource('s3')
crawler = WebCrawler()
crawler.warmup()
S3_BUCKET_NAME = 'contentcrawler'  # Replace with your S3 bucket name

# Helper function to get the main word from the URL
def get_main_word_from_url(base_url):
    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc
    main_word = domain.split('.')[0]
    return main_word

# Helper function to find all subpages from a base URL
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

# Route to crawl website content and upload to S3
@app.route('/crawl_website', methods=['POST'])
def crawl_website():
    base_url = request.form.get('url')
    if not base_url:
        return jsonify({'error': 'URL is required'}), 400

    # Get subpages and main word
    subpages = find_subpages(base_url)
    main_word = get_main_word_from_url(base_url)
    data = []

    # Create and store crawler results in a text file
    with open('crawler_results.txt', 'w', encoding='utf-8') as f_text:
        for i in subpages:
            result = crawler.run(url=i)
            if result.markdown != "None":
                f_text.write(f"URL: {i}\nContent:\n{result.markdown}\n" + "="*50 + "\n")
                data.append({
                    "url": i,
                    "content": result.markdown
                })

    # Upload text file to S3
    s3.meta.client.upload_file('crawler_results.txt', S3_BUCKET_NAME, f"{main_word}.txt")

    # Save JSON version of the data if needed
    with open('crawler_results.json', 'w', encoding='utf-8') as f_json:
        json.dump(data, f_json, indent=4)

    return jsonify({'status': 'success', 'message': 'Crawling completed and data saved to S3'})


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/get_videos', methods=['POST'])
def get_videos():
    channel_url = request.form.get('channel_url')
    
    if not channel_url:
        return jsonify({'error': 'YouTube channel URL is required'}), 400
    
    videos = fetch_latest_videos(channel_url)
    return render_template('videos_overview.html', videos=videos, channel_url=channel_url)

@app.route('/delete_video', methods=['POST'])
def delete_video():
    data = request.get_json()  # Extract JSON data from request
    video_id = data.get('video_id')  # Safely access the video_id key
    
    if not video_id:
        return jsonify({'success': False, 'error': 'Video ID not provided'}), 400
    
    # Perform any necessary deletion logic here
    # For now, we just return success as a demo
    return jsonify({'success': True})

@app.route('/transcribe_remaining', methods=['POST'])
def transcribe_remaining():
    data = request.get_json()
    remaining_video_ids = data.get('video_ids', [])
    channel_id = data.get('channel_id')

    if not remaining_video_ids:
        return 'No videos to transcribe', 400

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

    return render_template('videos.html', videos=videos)


@app.route('/video_details/<video_id>', methods=['GET'])
def video_details(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    audio_file = download_youtube_audio(youtube_url)
    if not audio_file:
        return 'Error downloading video audio', 500

    wav_file = convert_mp3_to_wav(audio_file)
    if not wav_file:
        return 'Error converting to WAV', 500

    processed_wav = preprocess_audio(wav_file)
    transcription = transcribe_audio_to_text_with_timestamps(processed_wav)
    summary = summarize_transcription(transcription)

    # Save transcription and summary to a file
    save_transcription_and_summary(video_id, transcription, summary)

    video = {
        'videoId': video_id,
        'transcription': transcription,
        'summary': summary,
        'title': f'Video {video_id}'
    }

    return render_template('videos.html', videos=[video])


@app.route('/translate/<video_id>', methods=['POST'])
def translate_video(video_id):
    target_language = request.form.get('target_language', 'es')
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    audio_file = download_youtube_audio(youtube_url)
    if not audio_file:
        return jsonify({"error": "Failed to download audio"}), 500

    wav_file = convert_mp3_to_wav(audio_file)
    if not wav_file:
        return jsonify({"error": "Failed to process audio"}), 500

    processed_wav = preprocess_audio(wav_file)
    transcription = transcribe_audio_to_text_with_timestamps(processed_wav)
    translation = translate_transcription_text(transcription, target_language)

    # Save transcription and translation (as summary) to a file
    save_transcription_and_summary(video_id, transcription, translation)

    if translation:
        return jsonify({"transcription": translation})
    else:
        return jsonify({"error": "Failed to translate"}), 500


def translate_transcription_text(transcription, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    try:
        return translator.translate(transcription)
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return None

if __name__ == "__main__":
    app.run(debug=True)