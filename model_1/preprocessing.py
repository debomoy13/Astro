import numpy as np
import scipy.signal as signal
from model_1.config import Config

class LightCurvePreprocessor:
    """
    Utility class for light curve preprocessing tasks including NaN interpolation,
    outlier clipping, normalization, linear detrending, and smoothing.
    """
    
    @staticmethod
    def handle_missing_values(flux, method='interpolate'):
        """
        Handles missing values (NaNs) in the light curve.
        Methods:
        - 'interpolate': linear interpolation (default)
        - 'median': fill with median
        - 'zero': fill with zeros
        """
        flux = np.array(flux, dtype=float)
        nans = np.isnan(flux)
        if not np.any(nans):
            return flux
        
        if method == 'interpolate':
            x = np.arange(len(flux))
            flux[nans] = np.interp(x[nans], x[~nans], flux[~nans])
        elif method == 'median':
            median_val = np.nanmedian(flux)
            flux[nans] = median_val
        elif method == 'zero':
            flux[nans] = 0.0
        return flux

    @staticmethod
    def normalize_flux(flux, method='zscore'):
        """
        Normalizes the flux values of the light curve.
        Methods:
        - 'zscore': subtract mean and divide by std (default)
        - 'median': divide by median and subtract 1.0 (relative flux centered at 0)
        - 'minmax': scale between 0 and 1
        """
        if method == 'zscore':
            std_val = np.std(flux)
            if std_val == 0:
                return flux
            return (flux - np.mean(flux)) / std_val
        elif method == 'median':
            median_val = np.median(flux)
            if median_val == 0:
                return flux
            return flux / median_val - 1.0
        elif method == 'minmax':
            min_val = np.min(flux)
            max_val = np.max(flux)
            if max_val - min_val == 0:
                return flux
            return (flux - min_val) / (max_val - min_val)
        return flux

    @staticmethod
    def remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2):
        """
        Applies iterative sigma clipping to remove extreme outliers and cosmic ray spikes.
        """
        clipped_flux = flux.copy()
        for _ in range(iters):
            mean = np.mean(clipped_flux)
            std = np.std(clipped_flux)
            if std == 0:
                break
            bad_idx = np.abs(clipped_flux - mean) > sigma * std
            clipped_flux[bad_idx] = mean
        return clipped_flux

    @staticmethod
    def remove_stellar_variability(flux, window_size=101):
        """
        Removes long-term stellar variability by subtracting a heavily median-filtered curve.
        """
        if window_size % 2 == 0:
            window_size += 1
        trend = signal.medfilt(flux, window_size)
        return flux - trend

    @staticmethod
    def savitzky_golay_filter(flux, window_size=15, polyorder=2):
        """
        Applies Savitzky-Golay filter to smooth high-frequency noise.
        """
        if window_size % 2 == 0:
            window_size += 1
        if len(flux) <= window_size:
            return flux
        return signal.savgol_filter(flux, window_size, polyorder)

    @staticmethod
    def detrend_linear(time, flux):
        """
        Applies linear detrending by subtracting a linear regression fit.
        """
        # Exclude NaNs from the fit if any remain (should be handled already)
        mask = ~np.isnan(time) & ~np.isnan(flux)
        if np.sum(mask) < 2:
            return flux
        slope, intercept = np.polyfit(time[mask], flux[mask], 1)
        linear_trend = slope * time + intercept
        return flux - linear_trend

    @classmethod
    def preprocess(cls, time, flux, config=Config, norm_method=None, detrend=True):
        """
        Sequentially runs preprocessing steps on the light curve:
        1. Gaps / NaNs interpolation
        2. Cosmic outlier rejection via sigma clipping
        3. Optional linear detrending
        4. Flattening slow stellar variability (median filter subtraction)
        5. Normalization
        6. High-frequency smoothing via Savitzky-Golay filter
        """
        # Ensure array
        time = np.array(time, dtype=float)
        flux = np.array(flux, dtype=float)
        
        # 1. Gaps / NaNs
        flux = cls.handle_missing_values(flux, method='interpolate')
        
        # 2. Outlier rejection
        flux = cls.remove_outliers_sigma_clipping(
            flux, 
            sigma=config.SIGMA_CLIPPING_SIGMA, 
            iters=config.SIGMA_CLIPPING_ITERS
        )
        
        # 3. Optional Linear Detrending
        if detrend:
            flux = cls.detrend_linear(time, flux)
            
        # 4. Stellar Variability removal
        flux = cls.remove_stellar_variability(
            flux, 
            window_size=config.MEDIAN_FILTER_WINDOW
        )
        
        # 5. Normalization
        norm_method = norm_method or config.NORM_METHOD
        flux = cls.normalize_flux(flux, method=norm_method)
        
        # 6. Savitzky-Golay smoothing
        flux = cls.savitzky_golay_filter(
            flux, 
            window_size=config.SG_WINDOW_SIZE, 
            polyorder=config.SG_POLYORDER
        )
        
        return flux

    @staticmethod
    def align_sequence_length(flux, target_len=2000, method='edge'):
        """
        Standardizes sequence dimensions by symmetrically cropping or padding.
        methods:
        - 'edge': pad with edge values (default)
        - 'zero': pad with zeros
        - 'symmetric': symmetric crop/pad
        """
        curr_len = len(flux)
        if curr_len > target_len:
            # Crop symmetrically from center
            start = (curr_len - target_len) // 2
            return flux[start:start + target_len]
        elif curr_len < target_len:
            # Pad with edge/zero values
            pad_width = target_len - curr_len
            pad_mode = 'edge' if method == 'edge' else 'constant'
            constant_values = 0.0 if method == 'zero' else 0.0
            
            if pad_mode == 'constant':
                return np.pad(flux, (0, pad_width), mode=pad_mode, constant_values=constant_values)
            else:
                return np.pad(flux, (0, pad_width), mode=pad_mode)
        return flux
