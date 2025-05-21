import requests
import json
from datetime import datetime, timedelta

# NASA API key
API_KEY = "n1X6pxLWkuXl8GTKlhrbAwLiEbdF9xEDXmBCc1wl"

def get_earth_imagery():
    """Get Earth imagery data from NASA's EPIC API for the previous 3 days"""
    print("\nEarth Polychromatic Imaging Camera (EPIC) Data for Previous 3 Days:")
    for i in range(1, 4):  # 1, 2, 3 days ago
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        url = f"https://api.nasa.gov/EPIC/api/natural/date/{date_str}?api_key={API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    print(f"\nDate: {date_str}")
                    for img in data:
                        print(f"  Image: https://epic.gsfc.nasa.gov/archive/natural/{date_str}/png/{img.get('image')}.png")
                        print(f"  Caption: {img.get('caption')}")
                        print(f"  Time: {img.get('date')}")
                        print("---")
                else:
                    print(f"No images available for {date_str}")
            else:
                print(f"Error: EPIC API request failed for {date_str} with status code {response.status_code}")
        except Exception as e:
            print(f"Error fetching EPIC data for {date_str}: {str(e)}")

def get_global_temperature():
    """Get global temperature anomaly data from NASA's GISTEMP API"""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("\nGlobal Temperature Anomaly Data (GISTEMP):")
            # Print the last 5 lines of data (most recent)
            lines = response.text.strip().split('\n')
            recent_data = lines[-5:]
            print("Recent Global Temperature Anomalies (degrees Celsius):")
            for line in recent_data:
                if line.strip() and not line.startswith('Year'):
                    print(line)
        else:
            print(f"Error: GISTEMP API request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching temperature data: {str(e)}")

def get_air_quality_data():
    """Get air quality data from NASA's NEO API"""
    # Example coordinates for a major city (New York)
    lat = 40.7128
    lon = -74.0060
    
    url = f"https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&date=2024-01-01&api_key={API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print("\nEarth Observation Data:")
            print(f"Date: {data.get('date')}")
            print(f"Image URL: {data.get('url')}")
            print("Note: This is a sample of Earth observation data. For detailed air quality data,")
            print("please visit NASA's Earth Observation Data Portal: https://earthdata.nasa.gov/")
        else:
            print(f"Error: Earth Observation API request failed with status code {response.status_code}")
    except Exception as e:
        print(f"Error fetching Earth observation data: {str(e)}")

def get_climate_data():
    """Get climate data from NASA's Climate API"""
    print("\nClimate Data Resources:")
    print("1. NASA's Global Climate Change: https://climate.nasa.gov/")
    print("2. NASA's Earth Observatory: https://earthobservatory.nasa.gov/")
    print("3. NASA's Earth Data: https://earthdata.nasa.gov/")
    print("\nKey Climate Indicators:")
    print("- Global Temperature Rise")
    print("- Sea Level Rise")
    print("- Arctic Sea Ice Decline")
    print("- Carbon Dioxide Levels")
    print("- Ocean Acidification")

def main():
    print("Fetching NASA Sustainability Data...")
    get_earth_imagery()
    get_global_temperature()
    get_air_quality_data()
    get_climate_data()

if __name__ == "__main__":
    main() 