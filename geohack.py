#!/usr/bin/env python3
import asyncio
import json

from cachetools import LRUCache
from geopy.geocoders import Nominatim

# Initialize cache and event loop
cache = LRUCache(maxsize=100)
LANGUAGE = 'en'
loop = asyncio.get_event_loop()

# Define URLs to intercept
METADATA_URL = 'https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata'
GEOGUESSR_API_URL = 'https://www.geoguessr.com/api/v3/games'

async def display_location(location):
    """Display formatted address details from the location object."""
    if not location or not location.raw.get('address'):
        print("No valid address data found.")
        return
    address = location.raw['address']
    for key, value in sorted(address.items()):
        # Filter out less useful address components
        if 'ISO3166' not in key and key not in ['postcode', 'country_code']:
            key = ' '.join(key.split('_'))
            print(f'\033[1m{key.title()}\033[0m: {value.title()}')
    print('\n')

async def get_location(coordinates):
    """Retrieve location details from coordinates, using cache if available."""
    trimmed_coordinates = (round(coordinates[0], 1), round(coordinates[1], 1))
    cached_location = cache.get(trimmed_coordinates)

    if cached_location:
        print("Using cached location data.")
        return cached_location, False

    geolocator = Nominatim(user_agent='Geoguessr')
    try:
        location = await loop.run_in_executor(
            None,
            lambda: geolocator.reverse(coordinates, language='en', addressdetails=True),
        )
    except Exception as e:
        print(f'Error with Nominatim API: {e}')
        return None, False

    cache[trimmed_coordinates] = location
    return location, False

async def response(flow):
    """Process HTTP responses to extract and display location data."""
    url = flow.request.pretty_url
    if flow.response.status_code == 200 and flow.request.method in ['GET', 'POST'] and flow.response.content:
        # Handle Geoguessr game API
        if GEOGUESSR_API_URL in url:
            print("Geoguessr game API response received")
            try:
                response_content = json.loads(flow.response.content)
                # Adjust the path based on actual response structure
                # Example path, may need adjustment
                lat = response_content.get('rounds', [{}])[-1].get('lat')
                lng = response_content.get('rounds', [{}])[-1].get('lng')
                if lat is not None and lng is not None:
                    print(f"Extracted coordinates from Geoguessr API: ({lat}, {lng})")
                    location, cache_hit = await get_location((lat, lng))
                    if location and not cache_hit:
                        await display_location(location)
                else:
                    print("No coordinates found in Geoguessr API response")
                    print("Response content:", json.dumps(response_content, indent=2))
            except (KeyError, TypeError, IndexError) as e:
                print(f"Error extracting coordinates from Geoguessr API: {e}")
                print("Response content:", json.dumps(response_content, indent=2))

        # Fallback: Handle Google Maps GetMetadata
        elif url == METADATA_URL and flow.request.method == 'POST':
            print("GetMetadata response received")
            try:
                response_content = json.loads(flow.response.content)
                lat = response_content[1][0][5][0][1][0][2]
                lng = response_content[1][0][5][0][1][0][3]
                print(f"Extracted coordinates from GetMetadata: ({lat}, {lng})")
                location, cache_hit = await get_location((lat, lng))
                if location and not cache_hit:
                    await display_location(location)
            except (IndexError, KeyError, TypeError) as e:
                print(f"Error extracting coordinates from GetMetadata: {e}")
                print("Response content:", json.dumps(response_content, indent=2))

# Ensure the script can be run with mitmproxy
if __name__ == "__main__":
    print("Script loaded. Start mitmproxy with this script to intercept traffic.")
