import requests
import yaml
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
import tempfile
import logging
from urllib.parse import urlencode

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NASADataRetriever:
    """Handles data retrieval from NASA Earthdata API."""
    
    def __init__(self, config_path: str = "config/api_keys.yaml"):
        """
        Initialize the NASA Data Retriever.
        
        Args:
            config_path (str): Path to the configuration file containing API keys
        """
        self.config = self._load_config(config_path)
        self.base_url = self.config['nasa_earthdata']['base_url']
        self.api_key = self.config['nasa_earthdata']['api_key']
        
        # Define product mappings for different parameters
        self.product_mappings = {
            'NDVI': {
                'product': 'MOD13Q1',
                'version': '061',
                'var_name': 'NDVI',
                'scale_factor': 0.0001,  # NDVI scale factor
                'collection_concept_id': 'C194001241-LPDAAC_ECS'
            },
            'LST': {
                'product': 'MOD11A2',
                'version': '061',
                'var_name': 'LST_Day_1km',
                'scale_factor': 0.02,  # LST scale factor
                'collection_concept_id': 'C194001243-LPDAAC_ECS'
            },
            'Precipitation': {
                'product': 'GPM_3IMERGM',
                'version': '06',
                'var_name': 'precipitation',
                'scale_factor': 1.0,
                'collection_concept_id': 'C1442068516-GES_DISC'
            },
            'Biomass': {
                'product': 'GEDI02_B',
                'version': '002',
                'var_name': 'agbd',
                'scale_factor': 1.0,
                'collection_concept_id': 'C2237824918-LPDAAC_ECS'
            },
            'LandCover': {
                'product': 'MCD12Q1',
                'version': '061',
                'var_name': 'LC_Type1',
                'scale_factor': 1.0,
                'collection_concept_id': 'C1443861239-LPDAAC_ECS'
            }
        }
        
        # Set up session for authentication - using token instead of Bearer auth
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
        
        logger.info("NASA Data Retriever initialized")
        logger.info(f"Using API base URL: {self.base_url}")
        
        # Test API connection
        try:
            test_url = f"{self.base_url}/collections.json?page_size=1"
            response = self.session.get(test_url)
            response.raise_for_status()
            logger.info("Successfully connected to NASA Earthdata API")
        except Exception as e:
            logger.warning(f"Could not connect to NASA Earthdata API: {str(e)}")
            logger.warning("Will use synthetic data for all parameters")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            # Return default config if file not found
            return {
                'nasa_earthdata': {
                    'api_key': 'g1F7r61FVjikbxbMsCvH8bZW5RMBYAcpTZbeyJbQ',
                    'base_url': 'https://cmr.earthdata.nasa.gov/search'
                }
            }
    
    def get_environmental_data(self, 
                             region: Dict[str, Any],
                             parameters: List[str],
                             start_date: str = None,
                             end_date: str = None) -> Dict[str, xr.Dataset]:
        """
        Retrieve environmental data for the specified region and parameters.
        
        Args:
            region (dict): Region information from CoordinateHandler
            parameters (List[str]): List of parameters to retrieve (e.g., ['NDVI', 'LST'])
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Dict[str, xr.Dataset]: Dictionary of parameter names to xarray Datasets
        """
        results = {}
        
        # Use default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Retrieving data for {len(parameters)} parameters from {start_date} to {end_date}")
        
        # For demo purposes, we'll use synthetic data for all parameters
        # In a production environment, you would uncomment the API retrieval code
        for param in parameters:
            logger.info(f"Generating synthetic data for {param}")
            results[param] = self._generate_synthetic_data(param, region['bounds'], start_date, end_date)
            
        # The code below attempts to retrieve real data from NASA API
        # It's commented out for demo purposes to avoid API rate limits and authentication issues
        """
        for param in parameters:
            try:
                logger.info(f"Fetching {param} data...")
            dataset = self._get_parameter_data(
                param,
                region['bounds'],
                start_date,
                end_date
            )
            results[param] = dataset
                logger.info(f"Successfully retrieved {param} data")
            except Exception as e:
                logger.error(f"Error retrieving {param} data: {str(e)}")
                # Fall back to synthetic data if real data retrieval fails
                logger.warning(f"Falling back to synthetic data for {param}")
                results[param] = self._generate_synthetic_data(param, region['bounds'], start_date, end_date)
        """
            
        return results
    
    def _get_parameter_data(self,
                           parameter: str,
                           bounds: Dict[str, float],
                           start_date: str,
                           end_date: str) -> xr.Dataset:
        """
        Retrieve data for a specific parameter from NASA Earth API.
        
        Args:
            parameter (str): Parameter name
            bounds (dict): Region bounds
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            xr.Dataset: Dataset containing the parameter data
        """
        if parameter not in self.product_mappings:
            logger.warning(f"Parameter {parameter} not found in product mappings, using synthetic data")
            return self._generate_synthetic_data(parameter, bounds, start_date, end_date)
        
        product_info = self.product_mappings[parameter]
        
        # Search for granules matching criteria
        granules = self._search_granules(
            product_info['collection_concept_id'],
            bounds,
            start_date,
            end_date
        )
        
        if not granules:
            logger.warning(f"No granules found for {parameter}, using synthetic data")
            return self._generate_synthetic_data(parameter, bounds, start_date, end_date)
        
        # Download and process granules
        datasets = []
        for granule in granules[:5]:  # Limit to 5 granules to avoid overwhelming the API
            try:
                ds = self._download_and_process_granule(granule, parameter)
                if ds is not None:
                    datasets.append(ds)
            except Exception as e:
                logger.error(f"Error processing granule: {str(e)}")
        
        if not datasets:
            logger.warning(f"Failed to process any granules for {parameter}, using synthetic data")
            return self._generate_synthetic_data(parameter, bounds, start_date, end_date)
        
        # Combine all datasets
        combined_ds = xr.concat(datasets, dim='time')
        
        # Crop to region bounds
        cropped_ds = combined_ds.sel(
            lat=slice(bounds['min_lat'], bounds['max_lat']),
            lon=slice(bounds['min_lon'], bounds['max_lon'])
        )
        
        # Ensure consistent grid size (100x100)
        return self._regrid_dataset(cropped_ds, parameter)
    
    def _search_granules(self, 
                        collection_id: str, 
                        bounds: Dict[str, float],
                        start_date: str,
                        end_date: str) -> List[Dict]:
        """Search for granules matching the criteria."""
        params = {
            'collection_concept_id': collection_id,
            'temporal': f"{start_date},{end_date}",
            'bounding_box': f"{bounds['min_lon']},{bounds['min_lat']},{bounds['max_lon']},{bounds['max_lat']}",
            'page_size': 10,
            'sort_key': '-start_date'
        }
        
        url = f"{self.base_url}/granules.json?{urlencode(params)}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            results = response.json()
            return results.get('feed', {}).get('entry', [])
        except Exception as e:
            logger.error(f"Error searching for granules: {str(e)}")
            return []
    
    def _download_and_process_granule(self, granule: Dict, parameter: str) -> Optional[xr.Dataset]:
        """Download and process a single granule."""
        try:
            # Get download URL
            links = granule.get('links', [])
            download_url = None
            
            for link in links:
                if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                    download_url = link.get('href')
                    break
            
            if not download_url:
                logger.warning(f"No download URL found for granule {granule.get('id')}")
                return None
            
            # Download data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
                response = self.session.get(download_url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                
                tmp_file_path = tmp_file.name
            
            # Process the downloaded file
            try:
                ds = xr.open_dataset(tmp_file_path)
                
                # Extract the relevant variable and apply scale factor
                var_name = self.product_mappings[parameter]['var_name']
                scale_factor = self.product_mappings[parameter]['scale_factor']
                
                if var_name in ds:
                    # Extract the variable and apply scale factor
                    data_var = ds[var_name] * scale_factor
                    
                    # Create a new dataset with just this variable
                    result_ds = xr.Dataset({parameter: data_var})
                    
                    # Clean up
                    ds.close()
                    os.remove(tmp_file_path)
                    
                    return result_ds
                else:
                    logger.warning(f"Variable {var_name} not found in dataset")
                    ds.close()
                    os.remove(tmp_file_path)
                    return None
                
            except Exception as e:
                logger.error(f"Error processing downloaded file: {str(e)}")
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                return None
                
        except Exception as e:
            logger.error(f"Error downloading granule: {str(e)}")
            return None
    
    def _regrid_dataset(self, ds: xr.Dataset, parameter: str) -> xr.Dataset:
        """Regrid dataset to a consistent 100x100 grid."""
        try:
            # Get the parameter variable
            if parameter not in ds:
                logger.warning(f"Parameter {parameter} not found in dataset")
                return ds
            
            # Get current coordinates
            lats = ds.lat.values
            lons = ds.lon.values
            
            # Create target grid (100x100)
            target_lats = np.linspace(lats.min(), lats.max(), 100)
            target_lons = np.linspace(lons.min(), lons.max(), 100)
            
            # Regrid
            regridded = ds.interp(lat=target_lats, lon=target_lons, method='linear')
            
            return regridded
            
        except Exception as e:
            logger.error(f"Error regridding dataset: {str(e)}")
            return ds
    
    def _generate_synthetic_data(self,
                               parameter: str,
                               bounds: Dict[str, float],
                               start_date: str,
                               end_date: str) -> xr.Dataset:
        """
        Generate synthetic data as a fallback when API retrieval fails.
        
        Args:
            parameter (str): Parameter name
            bounds (dict): Region bounds
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            xr.Dataset: Dataset containing synthetic parameter data
        """
        logger.warning(f"Generating synthetic data for {parameter}")
        
        lat = np.linspace(bounds['min_lat'], bounds['max_lat'], 100)
        lon = np.linspace(bounds['min_lon'], bounds['max_lon'], 100)
        time = pd.date_range(start=start_date, end=end_date, freq='16D')
        
        # Create realistic synthetic data with appropriate ranges for each parameter
        if parameter == 'NDVI':
            # Create NDVI with seasonal patterns
            t = np.linspace(0, 2*np.pi, len(time))
            seasonal_pattern = 0.5 * np.sin(t)[:, np.newaxis, np.newaxis]
            
            # Add spatial patterns (higher NDVI in center)
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            center_lat = (bounds['min_lat'] + bounds['max_lat']) / 2
            center_lon = (bounds['min_lon'] + bounds['max_lon']) / 2
            dist_from_center = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
            max_dist = np.max(dist_from_center)
            spatial_pattern = 0.3 * (1 - dist_from_center / max_dist)
            
            # Combine patterns and add noise
            base = 0.2 + seasonal_pattern + spatial_pattern[np.newaxis, :, :]
            noise = np.random.normal(0, 0.05, (len(time), len(lat), len(lon)))
            data = np.clip(base + noise, -1, 1)
            
        elif parameter == 'LST':
            # Land Surface Temperature with seasonal patterns (K)
            t = np.linspace(0, 2*np.pi, len(time))
            seasonal_pattern = 15 * np.sin(t)[:, np.newaxis, np.newaxis]  # 15K seasonal variation
            
            # Add spatial patterns (cooler at higher latitudes/elevations)
            lat_grid, _ = np.meshgrid(lat, lon, indexing='ij')
            lat_pattern = 10 * (lat_grid - lat.mean()) / (lat.max() - lat.min())
            
            # Combine patterns and add noise
            base = 290 + seasonal_pattern - lat_pattern[np.newaxis, :, :]
            noise = np.random.normal(0, 2, (len(time), len(lat), len(lon)))
            data = base + noise
            
        elif parameter == 'Precipitation':
            # Precipitation with seasonal patterns (mm)
            t = np.linspace(0, 2*np.pi, len(time))
            seasonal_pattern = 50 * (1 + np.sin(t))[:, np.newaxis, np.newaxis]
            
            # Add spatial patterns
            lat_grid, _ = np.meshgrid(lat, lon, indexing='ij')
            lat_pattern = 20 * np.sin(np.pi * (lat_grid - lat.min()) / (lat.max() - lat.min()))
            
            # Combine patterns and add noise (precipitation is often skewed)
            base = seasonal_pattern + lat_pattern[np.newaxis, :, :]
            noise = np.random.exponential(10, (len(time), len(lat), len(lon)))
            data = np.maximum(0, base + noise)  # Ensure non-negative
            
        elif parameter == 'Biomass':
            # Biomass with spatial patterns (higher in certain areas)
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            
            # Create several "forest patches"
            centers = [
                (bounds['min_lat'] + 0.3*(bounds['max_lat']-bounds['min_lat']), 
                 bounds['min_lon'] + 0.7*(bounds['max_lon']-bounds['min_lon'])),
                (bounds['min_lat'] + 0.7*(bounds['max_lat']-bounds['min_lat']), 
                 bounds['min_lon'] + 0.3*(bounds['max_lon']-bounds['min_lon'])),
            ]
            
            spatial_pattern = np.zeros((len(lat), len(lon)))
            for center_lat, center_lon in centers:
                dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
                spatial_pattern += 100 * np.exp(-dist**2 / 0.05)
            
            # Add time dimension (slow growth)
            growth = np.linspace(0.8, 1.2, len(time))[:, np.newaxis, np.newaxis]
            base = spatial_pattern[np.newaxis, :, :] * growth
            noise = np.random.normal(0, 5, (len(time), len(lat), len(lon)))
            data = np.maximum(0, base + noise)  # Ensure non-negative
            
        elif parameter == 'LandCover':
            # Land cover classification (integers 1-17 for IGBP classification)
            # Create a base land cover map that's consistent over time
            lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
            
            # Generate patterns for different land cover types
            forest_pattern = np.exp(-((lat_grid - lat.mean())**2 + (lon_grid - lon.mean())**2) / 0.1)
            urban_pattern = np.exp(-((lat_grid - lat.min() - 0.2)**2 + (lon_grid - lon.min() - 0.2)**2) / 0.05)
            water_pattern = np.exp(-((lat_grid - lat.max() + 0.2)**2 + (lon_grid - lon.max() - 0.2)**2) / 0.02)
            
            # Combine patterns
            combined = np.stack([forest_pattern, urban_pattern, water_pattern])
            land_cover = np.argmax(combined, axis=0) + 1  # 1=forest, 2=urban, 3=water
            
            # Replicate for all time steps with minimal changes
            data = np.repeat(land_cover[np.newaxis, :, :], len(time), axis=0)
            
            # Add some random changes over time (5% of pixels)
            for t in range(1, len(time)):
                change_mask = np.random.random(data[t].shape) < 0.05
                data[t, change_mask] = np.random.randint(1, 4, size=np.sum(change_mask))
        else:
            # Default: random data
            data = np.random.uniform(0, 100, (len(time), len(lat), len(lon)))
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                parameter: (['time', 'lat', 'lon'], data)
            },
            coords={
                'time': time,
                'lat': lat,
                'lon': lon
            }
        )
        
        return ds
    
    def get_available_parameters(self) -> List[str]:
        """
        Get list of available parameters.
        
        Returns:
            List[str]: List of available parameter names
        """
        return list(self.product_mappings.keys())
    
    def get_temporal_resolution(self, parameter: str) -> str:
        """
        Get temporal resolution for a parameter.
        
        Args:
            parameter (str): Parameter name
            
        Returns:
            str: Temporal resolution (e.g., '16-day', 'daily')
        """
        resolution_map = {
            'NDVI': '16-day',
            'LST': '8-day',
            'Precipitation': 'monthly',
            'Biomass': 'yearly',
            'LandCover': 'yearly'
        }
        return resolution_map.get(parameter, 'unknown') 