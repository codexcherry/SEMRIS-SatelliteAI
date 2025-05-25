"""
Coordinate Handler Module for managing region selection and coordinates.
"""

import numpy as np
from shapely.geometry import Point, box
from typing import Tuple, Optional, Union, Dict, List, Any

class CoordinateHandler:
    """
    Handles region selection and coordinate transformations.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Default parameters
        self.default_params = {
            'grid_size': 100,  # Number of points in lat/lon grid
            'min_radius': 1.0,  # Minimum radius in km
            'max_radius': 1000.0,  # Maximum radius in km
            'default_radius': 50.0  # Default radius in km
        }
        
        # Update with config values if provided
        if 'region_selection' in self.config:
            self.params = {
                **self.default_params,
                **self.config['region_selection']
            }
        else:
            self.params = self.default_params
        
        self.earth_radius = 6371  # Earth's radius in kilometers
        
    def select_region(
        self,
        lat: float,
        lon: float,
        radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Select a region around given coordinates.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Dictionary with region information
        """
        # Validate and adjust radius
        if radius_km is None:
            radius_km = self.params['default_radius']
        
        radius_km = np.clip(
            radius_km,
            self.params['min_radius'],
            self.params['max_radius']
        )
        
        # Calculate region bounds
        lat_range, lon_range = self._calculate_bounds(
            lat,
            lon,
            radius_km
        )
        
        # Create coordinate grids
        lat_grid, lon_grid = self._create_coordinate_grid(
            lat_range,
            lon_range
        )
        
        # Calculate distances from center
        distances = self._calculate_distances(
            lat_grid,
            lon_grid,
            lat,
            lon
        )
        
        # Create region mask
        region_mask = distances <= radius_km
        
        return {
            'center': {'lat': lat, 'lon': lon},
            'radius_km': radius_km,
            'bounds': {
                'min_lat': float(lat_range[0]),
                'max_lat': float(lat_range[1]),
                'min_lon': float(lon_range[0]),
                'max_lon': float(lon_range[1])
            },
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'mask': region_mask,
            'grid_size': self.params['grid_size']
        }
    
    def _calculate_bounds(
        self,
        lat: float,
        lon: float,
        radius_km: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate region bounds based on center and radius.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Tuple of (lat_range, lon_range)
        """
        # Approximate degrees per km at this latitude
        lat_deg_per_km = 1/111.0  # Roughly 111 km per degree
        lon_deg_per_km = 1/(111.0 * np.cos(np.radians(lat)))
        
        # Calculate ranges
        lat_range = (
            lat - radius_km * lat_deg_per_km,
            lat + radius_km * lat_deg_per_km
        )
        
        lon_range = (
            lon - radius_km * lon_deg_per_km,
            lon + radius_km * lon_deg_per_km
        )
        
        return lat_range, lon_range
    
    def _create_coordinate_grid(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grids for the region.
        
        Args:
            lat_range: (min_lat, max_lat)
            lon_range: (min_lon, max_lon)
            
        Returns:
            Tuple of (lat_grid, lon_grid)
        """
        # Create coordinate arrays
        lats = np.linspace(
            lat_range[0],
            lat_range[1],
            self.params['grid_size']
        )
        
        lons = np.linspace(
            lon_range[0],
            lon_range[1],
            self.params['grid_size']
        )
        
        # Create coordinate grids
        return np.meshgrid(lats, lons)
    
    def _calculate_distances(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        center_lat: float,
        center_lon: float
    ) -> np.ndarray:
        """
        Calculate distances from center point to all grid points.
        
        Args:
            lat_grid: Grid of latitudes
            lon_grid: Grid of longitudes
            center_lat: Center latitude
            center_lon: Center longitude
            
        Returns:
            Grid of distances in kilometers
        """
        # Convert to radians
        lat1 = np.radians(center_lat)
        lon1 = np.radians(center_lon)
        lat2 = np.radians(lat_grid)
        lon2 = np.radians(lon_grid)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (
            np.sin(dlat/2)**2 +
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371.0
        
        return c * r
    
    def get_grid_coordinates(
        self,
        region: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """
        Get list of all grid point coordinates.
        
        Args:
            region: Region dictionary from select_region
            
        Returns:
            List of coordinate dictionaries
        """
        lat_grid = region['lat_grid']
        lon_grid = region['lon_grid']
        mask = region['mask']
        
        coordinates = []
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                if mask[i, j]:
                    coordinates.append({
                        'lat': float(lat_grid[i, j]),
                        'lon': float(lon_grid[i, j])
                    })
        
        return coordinates
    
    def is_point_in_region(
        self,
        lat: float,
        lon: float,
        region: Dict[str, Any]
    ) -> bool:
        """
        Check if a point is within the region.
        
        Args:
            lat: Point latitude
            lon: Point longitude
            region: Region dictionary from select_region
            
        Returns:
            True if point is in region
        """
        distance = self._calculate_distances(
            np.array([[lat]]),
            np.array([[lon]]),
            region['center']['lat'],
            region['center']['lon']
        )[0, 0]
        
        return distance <= region['radius_km']
    
    def get_region_params(self) -> Dict[str, Any]:
        """
        Get current region selection parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def set_region_params(
        self,
        params: Dict[str, Any]
    ) -> None:
        """
        Update region selection parameters.
        
        Args:
            params: New parameter values
        """
        self.params.update(params)
    
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