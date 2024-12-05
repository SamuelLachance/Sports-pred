import os
import re
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


class Album:
    def __init__(self, url):
        self.url = url
        self.songs = []

    def download(self, path, num_workers=5):
        # Dynamically create a folder name based on the URL
        folder_name = sanitize_file_name(self.url.split('/')[-1])
        album_path = os.path.join(path, folder_name)

        # Ensure the download directory exists
        os.makedirs(album_path, exist_ok=True)

        # Create a thread pool for downloading files
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for song in self.songs:
                filename = sanitize_file_name(f"{song['name']}.mp3")
                file_path = os.path.join(album_path, filename)
                futures.append(executor.submit(download_file, song['download_url'], file_path))

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading file: {e}")

        print(f"Download complete. Files are stored in {album_path}")


def fetch_page(url):
    """Fetches a web page and returns the parsed HTML document."""
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {response.status_code}")
    return BeautifulSoup(response.text, "html.parser")


def parse_album(url):
    """Parses an album page and extracts song details."""
    album_page = fetch_page(url)
    album = Album(url)

    # Find all songs in the album
    rows = album_page.select("table#songlist tr")
    for row in rows:
        link = row.find("a")
        if link:
            song_url = f"https://downloads.khinsider.com{link['href']}"
            song_name = link.text.strip()
            download_url = get_download_link(song_url)
            if download_url:
                album.songs.append({
                    "url": song_url,
                    "name": song_name,
                    "download_url": download_url,
                })

    return album


def get_download_link(song_url):
    """Fetches the download link for a specific song."""
    song_page = fetch_page(song_url)
    link = song_page.find("a", href=re.compile(r'/soundtracks/'))
    if not link:
        raise Exception(f"Download link not found for {song_url}")
    return link['href']


def download_file(url, path):
    """Downloads a file from a URL and saves it locally."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")

    with open(path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded: {path}")


def sanitize_file_name(name):
    """Sanitizes a file name to remove invalid characters."""
    name = name.replace(":", "-")
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip()
    name = name.replace(" ", "_")
    return name


if __name__ == "__main__":
    # Prompt the user to enter the soundtrack URL
    soundtrack_url = input("Enter the soundtrack URL: ").strip()

    try:
        soundtrack = parse_album(soundtrack_url)
        soundtrack.download("./downloads", num_workers=5)
    except Exception as e:
        print(f"Error: {e}")
