import requests
import brotli

# Define the URL
url = "https://ozajxwcjhgjugnhluqcm.supabase.co/rest/v1/game-projections?select=*&date=eq.2024-10-17"

# Define headers
headers = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "accept-profile": "public",
    "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96YWp4d2NqaGdqdWduaGx1cWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTg2NzIxNDAsImV4cCI6MjAzNDI0ODE0MH0.V4rXuEbCloTCWeIAa-eCHcq9lvPi2mDhOMAaUL5oxGY",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96YWp4d2NqaGdqdWduaGx1cWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTg2NzIxNDAsImV4cCI6MjAzNDI0ODE0MH0.V4rXuEbCloTCWeIAa-eCHcq9lvPi2mDhOMAaUL5oxGY",
    "origin": "https://iceanalytics.ca",
    "referer": "https://iceanalytics.ca/",
    "sec-ch-ua": '"Chromium";v="129", "Not=A?Brand";v="8"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "x-client-info": "supabase-js-web/2.43.5"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    try:
        # If Brotli encoding is used, decode the content manually
        if response.headers.get('Content-Encoding') == 'br':
            decompressed_data = brotli.decompress(response.content)
            data = decompressed_data.decode('utf-8')
            print(data)
        else:
            # For other encodings (gzip, deflate) or no encoding, requests handles it automatically
            print(response.json())
    except requests.exceptions.JSONDecodeError:
        # If there's a JSONDecodeError, print the raw response content
        print("Failed to decode JSON. Raw content:")
        print(response.text)
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Response content:", response.text)
