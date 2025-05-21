import numpy as np
from shapely.geometry import Point, box
from typing import Tuple, Optional, Union

class CoordinateHandler:
    """Handles geographic coordinate operations and region selection."""
    
    def __init__(self):
        self.earth_radius = 6371  # Earth's radius in kilometers
        
    def select_region(self, 
                     lat: float, 
                     lon: float, 
                     radius_km: float) -> dict:
        """
        Select a circular region based on center coordinates and radius.
        
        Args:
            lat (float): Latitude of the center point
            lon (float): Longitude of the center point
            radius_km (float): Radius of the region in kilometers
            
        Returns:
            dict: Region information including bounds and geometry
        """
        # Calculate bounding box
        bounds = self._calculate_bounds(lat, lon, radius_km)
        
        # Create region geometry
        center_point = Point(lon, lat)
        region_box = box(bounds['min_lon'], bounds['min_lat'],
                        bounds['max_lon'], bounds['max_lat'])
        
        return {
            'center': {'lat': lat, 'lon': lon},
            'radius_km': radius_km,
            'bounds': bounds,
            'geometry': region_box,
            'center_point': center_point
        }
    
    def _calculate_bounds(self, 
                         lat: float, 
                         lon: float, 
                         radius_km: float) -> dict:
        """
        Calculate the bounding box for a circular region.
        
        Args:
            lat (float): Latitude of the center point
            lon (float): Longitude of the center point
            radius_km (float): Radius of the region in kilometers
            
        Returns:
            dict: Bounding box coordinates
        """
        # Convert radius to degrees (approximate)
        lat_deg = radius_km / self.earth_radius * (180 / np.pi)
        lon_deg = radius_km / (self.earth_radius * np.cos(np.radians(lat))) * (180 / np.pi)
        
        return {
            'min_lat': lat - lat_deg,
            'max_lat': lat + lat_deg,
            'min_lon': lon - lon_deg,
            'max_lon': lon + lon_deg
        }
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate if the given coordinates are within valid ranges.
        
        Args:
            lat (float): Latitude to validate
            lon (float): Longitude to validate
            
        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    def calculate_area(self, region: dict) -> float:
        """
        Calculate the area of the selected region in square kilometers.
        
        Args:
            region (dict): Region information from select_region
            
        Returns:
            float: Area in square kilometers
        """
        radius = region['radius_km']
        return np.pi * radius * radius 