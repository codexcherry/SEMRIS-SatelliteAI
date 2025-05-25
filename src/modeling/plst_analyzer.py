"""
Advanced Probabilistic Land Surface Temperature (PLST) Analysis Module.
Implements sophisticated multi-parameter analysis for environmental monitoring.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xarray as xr
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

class PLSTAnalyzer:
    """
    Advanced PLST Analyzer for environmental degradation risk assessment.
    Combines multiple parameters including LST, NDVI, precipitation, and soil moisture
    to generate comprehensive risk assessments.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Default analysis parameters
        self.default_params = {
            'n_components': 3,  # Number of principal components
            'contamination': 0.1,  # Expected proportion of anomalies
            'temporal_window': 30,  # Days for temporal analysis
            'confidence_threshold': 0.8  # Minimum confidence for patterns
        }
        
        # Update with config values if provided
        if 'analysis' in self.config:
            self.params = {
                **self.default_params,
                **self.config['analysis']
            }
        else:
            self.params = self.default_params
        
        # Initialize components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.params['n_components'])
        self.isolation_forest = IsolationForest(
            contamination=self.params['contamination'],
            random_state=42
        )
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.parameter_weights = {
            'LST': 0.35,
            'NDVI': 0.25,
            'Precipitation': 0.20,
            'SoilMoisture': 0.20
        }
    
    def calculate_lst_anomalies(
        self,
        lst_data: np.ndarray,
        baseline_period: Tuple[str, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate LST anomalies using advanced statistical methods.
        
        Args:
            lst_data: Time series of LST measurements
            baseline_period: Tuple of (start_date, end_date) for baseline
            
        Returns:
            Tuple of (anomalies, extreme_mask)
        """
        baseline_mean = np.mean(lst_data)
        baseline_std = np.std(lst_data)
        
        # Calculate standardized anomalies
        anomalies = (lst_data - baseline_mean) / baseline_std
        
        # Apply extreme value analysis
        threshold = 2.0  # Standard deviations
        extreme_mask = np.abs(anomalies) > threshold
        
        return anomalies, extreme_mask
    
    def analyze_vegetation_stress(
        self,
        ndvi_data: np.ndarray,
        lst_anomalies: np.ndarray
    ) -> np.ndarray:
        """
        Analyze vegetation stress using NDVI and LST anomalies.
        
        Args:
            ndvi_data: NDVI time series
            lst_anomalies: LST anomaly data
            
        Returns:
            Vegetation stress index
        """
        # Calculate NDVI trends
        ndvi_trend = np.gradient(ndvi_data)
        
        # Combine with LST anomalies
        stress_index = (
            -ndvi_trend * 0.6 +  # Negative NDVI trend indicates stress
            lst_anomalies * 0.4  # High LST anomalies indicate stress
        )
        
        return stress_index
    
    def calculate_drought_index(
        self,
        precipitation: np.ndarray,
        soil_moisture: np.ndarray,
        temperature: np.ndarray
    ) -> np.ndarray:
        """
        Calculate comprehensive drought index using multiple parameters.
        
        Args:
            precipitation: Precipitation data
            soil_moisture: Soil moisture data
            temperature: Temperature data
            
        Returns:
            Drought severity index
        """
        # Standardize inputs
        precip_z = stats.zscore(precipitation)
        soil_z = stats.zscore(soil_moisture)
        temp_z = stats.zscore(temperature)
        
        # Calculate drought index (modified SPEI)
        drought_index = (
            precip_z * 0.4 +
            soil_z * 0.35 +
            -temp_z * 0.25  # Higher temperatures increase drought severity
        )
        
        return drought_index
    
    def assess_degradation_risk(
        self,
        data: Dict[str, np.ndarray],
        temporal_window: int = 12
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Perform comprehensive degradation risk assessment.
        
        Args:
            data: Dictionary of parameter data arrays
            temporal_window: Analysis window in months
            
        Returns:
            Tuple of (risk_scores, component_indices)
        """
        # Ensure all arrays are 2D (time x features)
        processed_data = {}
        for param_name, param_data in data.items():
            if param_data.ndim > 2:
                n_samples = param_data.shape[0]
                processed_data[param_name] = param_data.reshape(n_samples, -1)
            elif param_data.ndim == 1:
                processed_data[param_name] = param_data.reshape(-1, 1)
            else:
                processed_data[param_name] = param_data
        
        # Calculate component indices
        lst_anomalies, extreme_lst = self.calculate_lst_anomalies(
            processed_data['LST'],
            baseline_period=('2020-01', '2022-12')
        )
        
        veg_stress = self.analyze_vegetation_stress(
            processed_data['NDVI'],
            lst_anomalies
        )
        
        drought_index = self.calculate_drought_index(
            processed_data['Precipitation'],
            processed_data.get('SoilMoisture', np.zeros_like(processed_data['Precipitation'])),
            processed_data['LST']
        )
        
        # Combine indices into final risk score
        risk_score = (
            lst_anomalies.mean(axis=1) * self.parameter_weights['LST'] +
            veg_stress.mean(axis=1) * self.parameter_weights['NDVI'] +
            drought_index.mean(axis=1) * (
                self.parameter_weights['Precipitation'] +
                self.parameter_weights['SoilMoisture']
            )
        )
        
        # Normalize to 0-1 range
        risk_score = (risk_score - np.min(risk_score)) / (np.max(risk_score) - np.min(risk_score))
        
        component_indices = {
            'lst_anomalies': lst_anomalies.mean(axis=1),
            'vegetation_stress': veg_stress.mean(axis=1),
            'drought_index': drought_index.mean(axis=1),
            'extreme_lst_events': extreme_lst.any(axis=1)
        }
        
        return risk_score, component_indices
    
    def generate_risk_report(
        self,
        risk_score: np.ndarray,
        component_indices: Dict[str, np.ndarray],
        region_metadata: Dict
    ) -> Dict:
        """
        Generate detailed risk assessment report.
        
        Args:
            risk_score: Overall risk scores
            component_indices: Component analysis results
            region_metadata: Metadata about the analyzed region
            
        Returns:
            Dictionary containing report data
        """
        report = {
            'summary': {
                'mean_risk': float(np.mean(risk_score)),
                'max_risk': float(np.max(risk_score)),
                'risk_trend': float(np.gradient(risk_score).mean()),
                'high_risk_areas_percent': float(np.mean(risk_score > 0.7) * 100)
            },
            'components': {
                'lst_anomalies': {
                    'mean': float(np.mean(component_indices['lst_anomalies'])),
                    'extreme_events_count': int(np.sum(component_indices['extreme_lst_events']))
                },
                'vegetation_stress': {
                    'mean': float(np.mean(component_indices['vegetation_stress'])),
                    'severe_stress_areas': float(np.mean(component_indices['vegetation_stress'] > 0.7) * 100)
                },
                'drought_severity': {
                    'mean': float(np.mean(component_indices['drought_index'])),
                    'severe_drought_areas': float(np.mean(component_indices['drought_index'] < -1.5) * 100)
                }
            },
            'recommendations': self._generate_recommendations(risk_score, component_indices),
            'metadata': {
                'region': region_metadata,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'version': '2.0.0'
            }
        }
        
        return report
    
    def _generate_recommendations(
        self,
        risk_score: np.ndarray,
        component_indices: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        Generate targeted recommendations based on analysis results.
        """
        recommendations = []
        
        # High LST anomalies recommendations
        if np.mean(component_indices['lst_anomalies']) > 1.5:
            recommendations.append({
                'category': 'Temperature Management',
                'priority': 'High',
                'action': 'Implement urban heat island mitigation measures',
                'details': 'Consider increasing green cover and reflective surfaces'
            })
        
        # Vegetation stress recommendations
        if np.mean(component_indices['vegetation_stress']) > 0.6:
            recommendations.append({
                'category': 'Vegetation Protection',
                'priority': 'High',
                'action': 'Enhance vegetation monitoring and protection',
                'details': 'Implement irrigation management and soil conservation'
            })
        
        # Drought management recommendations
        if np.mean(component_indices['drought_index']) < -1.0:
            recommendations.append({
                'category': 'Drought Management',
                'priority': 'Critical',
                'action': 'Activate drought response protocols',
                'details': 'Implement water conservation measures and monitoring'
            })
        
        return recommendations
    
    def analyze(
        self,
        input_data: Dict[str, Any],
        temporal_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform PLST analysis on environmental data.
        
        Args:
            input_data: Input data dictionary
            temporal_filter: Optional time window filter in days
            
        Returns:
            Analysis results
        """
        # Extract data
        data = input_data['data']
        location = input_data.get('location', {})
        timestamp = input_data.get('timestamp')
        
        # Prepare data arrays
        parameter_arrays = []
        parameter_names = []
        
        for param_name, param_data in data.items():
            if isinstance(param_data, np.ndarray):
                # Ensure 2D array (time x features)
                if param_data.ndim > 2:
                    n_samples = param_data.shape[0]
                    param_data = param_data.reshape(n_samples, -1)
                elif param_data.ndim == 1:
                    param_data = param_data.reshape(-1, 1)
                parameter_arrays.append(param_data)
                parameter_names.append(param_name)
        
        if not parameter_arrays:
            raise ValueError("No valid parameter arrays found in input data")
        
        # Stack arrays for analysis
        try:
            X = np.hstack(parameter_arrays)  # Stack horizontally to combine features
        except ValueError as e:
            raise ValueError(f"Error stacking parameter arrays: {e}. Check that all arrays have the same first dimension.")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Detect anomalies
        anomaly_scores = self.isolation_forest.fit_predict(X_pca)
        
        # Calculate risk scores
        risk_scores = self._calculate_risk_scores(
            X_pca,
            anomaly_scores
        )
        
        # Extract component indices
        component_indices = self._extract_component_indices(
            X_pca,
            parameter_names
        )
        
        # Generate temporal patterns if time information available
        temporal_patterns = None
        if timestamp is not None:
            temporal_patterns = self._analyze_temporal_patterns(
                risk_scores,
                timestamp,
                temporal_filter
            )
        
        # Prepare results
        results = {
            'risk_scores': risk_scores,
            'component_indices': component_indices,
            'explained_variance': list(self.pca.explained_variance_ratio_),
            'anomaly_scores': anomaly_scores.tolist(),
            'temporal_patterns': temporal_patterns,
            'metadata': {
                'timestamp': timestamp,
                'location': location,
                'parameters': parameter_names,
                'confidence': self._calculate_confidence(risk_scores)
            }
        }
        
        return results
    
    def _calculate_risk_scores(
        self,
        X_pca: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk scores from PCA and anomaly detection results.
        
        Args:
            X_pca: PCA transformed data
            anomaly_scores: Anomaly detection scores
            
        Returns:
            Array of risk scores
        """
        # Convert anomaly scores to probabilities
        anomaly_probs = 1 / (1 + np.exp(-np.abs(anomaly_scores)))
        
        # Calculate component magnitudes
        magnitudes = np.linalg.norm(X_pca, axis=1)
        
        # Normalize magnitudes
        magnitudes = (magnitudes - np.min(magnitudes)) / (
            np.max(magnitudes) - np.min(magnitudes)
        )
        
        # Combine scores
        risk_scores = 0.7 * magnitudes + 0.3 * anomaly_probs
        
        return risk_scores
    
    def _extract_component_indices(
        self,
        X_pca: np.ndarray,
        parameter_names: List[str]
    ) -> Dict[str, List[float]]:
        """
        Extract component indices for each parameter.
        
        Args:
            X_pca: PCA transformed data
            parameter_names: List of parameter names
            
        Returns:
            Dictionary of component indices
        """
        # Get component loadings
        loadings = self.pca.components_  # Shape: (n_components, n_features)
        
        # Calculate feature importances for each parameter
        indices = {}
        feature_count = loadings.shape[1] // len(parameter_names)
        
        for i, param_name in enumerate(parameter_names):
            # Calculate start and end indices for this parameter
            start_idx = i * feature_count
            end_idx = (i + 1) * feature_count
            
            # Get loadings for this parameter's features
            param_loadings = loadings[:, start_idx:end_idx]
            
            # Calculate importance score (sum of absolute loadings)
            importance = float(np.abs(param_loadings).sum(axis=0).mean())
            indices[param_name] = importance
        
        return indices
    
    def _analyze_temporal_patterns(
        self,
        risk_scores: np.ndarray,
        timestamp: str,
        temporal_filter: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal patterns in risk scores.
        
        Args:
            risk_scores: Array of risk scores
            timestamp: Current timestamp
            temporal_filter: Optional time window filter
            
        Returns:
            Dictionary of temporal patterns or None
        """
        try:
            # Convert timestamp to datetime
            current_time = pd.to_datetime(timestamp)
            
            # Create time index
            if temporal_filter:
                time_points = pd.date_range(
                    end=current_time,
                    periods=len(risk_scores),
                    freq=f"{temporal_filter}D"
                )
            else:
                time_points = pd.date_range(
                    end=current_time,
                    periods=len(risk_scores),
                    freq="D"
                )
            
            # Create time series
            ts = pd.Series(risk_scores, index=time_points)
            
            # Calculate trends
            trends = {
                'short_term': float(ts.diff().mean()),  # Daily change
                'long_term': float(  # Weekly change
                    ts.rolling(window=7).mean().diff().mean()
                )
            }
            
            # Detect seasonality
            if len(ts) >= 30:
                seasonal_corr = float(ts.autocorr(lag=30))  # Monthly pattern
            else:
                seasonal_corr = None
            
            return {
                'trends': trends,
                'seasonality': seasonal_corr,
                'time_range': {
                    'start': time_points[0].isoformat(),
                    'end': time_points[-1].isoformat()
                }
            }
            
        except Exception:
            return None
    
    def _calculate_confidence(
        self,
        risk_scores: np.ndarray
    ) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            risk_scores: Array of risk scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate confidence based on risk score distribution
        score_std = np.std(risk_scores)
        score_range = np.ptp(risk_scores)
        
        if score_range == 0:
            return 0.0
        
        # Higher confidence for more distinct patterns
        confidence = min(1.0, score_std / (0.1 * score_range))
        
        return float(confidence)
    
    def get_analysis_params(self) -> Dict[str, Any]:
        """
        Get current analysis parameters.
        
        Returns:
            Dictionary of analysis parameters
        """
        return self.params.copy()
    
    def set_analysis_params(
        self,
        params: Dict[str, Any]
    ) -> None:
        """
        Update analysis parameters.
        
        Args:
            params: New parameter values
        """
        self.params.update(params)
        
        # Update components if necessary
        if 'n_components' in params:
            self.pca = PCA(n_components=self.params['n_components'])
        if 'contamination' in params:
            self.isolation_forest = IsolationForest(
                contamination=self.params['contamination'],
                random_state=42
            ) 