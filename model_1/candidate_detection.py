import numpy as np
import scipy.signal as signal
from model_1.config import Config

try:
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

class TransitCandidateDetector:
    """
    Identifies exoplanet transit candidates using Box Least Squares (BLS) periodogram search.
    Provides support for extracting transit-centered windows (Option B).
    """

    @staticmethod
    def run_bls(time, flux, min_period=0.5, max_period=20.0, min_duration=0.05, max_duration=0.5):
        """
        Runs Box Least Squares (BLS) periodogram to find the strongest periodic transit signal.
        Returns:
            dict containing:
                'period': orbital period (days)
                'duration': transit duration (days)
                'depth': transit depth
                'time0': transit mid-time (epoch)
                'power': maximum BLS power
        """
        # Ensure input shapes are valid
        time = np.array(time, dtype=float)
        flux = np.array(flux, dtype=float)
        
        if len(time) < 10:
            return {'period': 1.0, 'duration': 0.1, 'depth': 0.0, 'time0': 0.0, 'power': 0.0}

        if ASTROPY_AVAILABLE:
            try:
                # Format variables for astropy
                t = time * u.day
                y = flux * u.dimensionless_unscaled
                
                # Setup duration grid
                durations = np.linspace(min_duration, max_duration, 5) * u.day
                
                # Initialize BLS model
                model = BoxLeastSquares(t, y)
                periodogram = model.autopower(durations, minimum_period=min_period, maximum_period=max_period)
                
                # Get index of maximum power
                max_idx = np.argmax(periodogram.power)
                period = float(periodogram.period[max_idx].value)
                time0 = float(periodogram.transit_time[max_idx].value)
                duration = float(periodogram.duration[max_idx].value)
                depth = float(periodogram.depth[max_idx].value)
                power = float(periodogram.power[max_idx].value)
                
                return {
                    'period': period,
                    'duration': duration,
                    'depth': depth,
                    'time0': time0,
                    'power': power
                }
            except Exception as e:
                # Fallback to custom heuristic if astropy BLS fails
                pass
        
        # Heuristic search fallback: Locate the lowest flux values to estimate midpoints
        # (Ideal for lightweight dry-runs / missing dependency environments)
        min_idx = np.argmin(flux)
        time0 = float(time[min_idx])
        
        # Estimate duration by finding width of transit dip at half minimum
        baseline = np.median(flux)
        depth = float(baseline - flux[min_idx])
        half_min = baseline - depth / 2.0
        
        # Find points below half-min around the transit center
        dip_indices = np.where(flux < half_min)[0]
        if len(dip_indices) > 0:
            duration = float(time[dip_indices[-1]] - time[dip_indices[0]])
        else:
            duration = 0.1
            
        return {
            'period': 5.0,  # Default fallback period
            'duration': max(0.01, duration),
            'depth': max(0.0, depth),
            'time0': time0,
            'power': 1.0
        }

    @staticmethod
    def extract_transit_window(time, flux, time0, period, window_size=200, align_method='edge'):
        """
        Phase-folds the light curve around the primary transit midpoint (epoch=time0, period=period)
        and crops a local window of length `window_size` centered exactly on the transit.
        
        This aligns the ingress/egress shapes of the transit directly in the sequence.
        """
        time = np.array(time, dtype=float)
        flux = np.array(flux, dtype=float)
        
        if len(time) == 0:
            return np.zeros(window_size)
            
        # Fold the times: relative phase centered on 0.0
        phase = ((time - time0 + 0.5 * period) % period) - 0.5 * period
        
        # Sort values by phase to get a continuous folded light curve
        sort_idx = np.argsort(phase)
        sorted_phase = phase[sort_idx]
        sorted_flux = flux[sort_idx]
        
        # Find index closest to phase 0.0 (the transit midpoint)
        mid_idx = np.argmin(np.abs(sorted_phase))
        
        # Slice window centered at phase 0.0
        half_w = window_size // 2
        start_idx = mid_idx - half_w
        end_idx = mid_idx + half_w
        
        # Handle boundaries using padding if the window overflows
        if start_idx >= 0 and end_idx < len(sorted_flux):
            window_flux = sorted_flux[start_idx:end_idx]
        else:
            # Crop/pad manually
            window_flux = []
            for i in range(start_idx, end_idx):
                if i < 0:
                    # Pad left side
                    val = sorted_flux[0] if align_method == 'edge' else 0.0
                elif i >= len(sorted_flux):
                    # Pad right side
                    val = sorted_flux[-1] if align_method == 'edge' else 0.0
                else:
                    val = sorted_flux[i]
                window_flux.append(val)
            window_flux = np.array(window_flux)
            
        return window_flux
