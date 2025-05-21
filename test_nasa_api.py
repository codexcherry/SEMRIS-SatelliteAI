import requests
import json

# NASA API key
API_KEY = "n1X6pxLWkuXl8GTKlhrbAwLiEbdF9xEDXmBCc1wl"

# NASA APOD API endpoint
url = f"https://api.nasa.gov/planetary/apod?api_key={API_KEY}"

try:
    # Make the API request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        print("Successfully retrieved data from NASA API!")
        print("\nAstronomy Picture of the Day:")
        print(f"Title: {data.get('title')}")
        print(f"Date: {data.get('date')}")
        print(f"Explanation: {data.get('explanation')}")
        print(f"Image URL: {data.get('url')}")
    else:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"An error occurred: {str(e)}") 