import numpy as np

class TimeSeriesAugmenter:
    """
    Implements astronomical data augmentations for 1D time series flux data.
    These augmentations are designed to avoid destroying critical features like
    ingress/egress shapes of exoplanet transits.
    """

    @staticmethod
    def add_gaussian_noise(flux, std_limit=0.03):
        """
        Adds random white Gaussian noise to simulate detector read-out noise or background star noise.
        """
        std = np.random.uniform(0.005, std_limit)
        noise = np.random.normal(0, std, size=flux.shape)
        return flux + noise

    @staticmethod
    def random_shift(flux, max_shift_frac=0.05):
        """
        Performs random circular translation shifts of the light curve.
        """
        shift_limit = int(max_shift_frac * len(flux))
        if shift_limit <= 0:
            return flux
        shift = np.random.randint(-shift_limit, shift_limit)
        return np.roll(flux, shift)

    @staticmethod
    def random_scale(flux, scale_range=(0.95, 1.05)):
        """
        Scales the relative amplitude of features by multiplying the flux centered around its baseline.
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        median_val = np.median(flux)
        return (flux - median_val) * scale + median_val

    @staticmethod
    def flux_jitter(flux, noise_level=0.01):
        """
        Applies a random minor high-frequency fluctuation (jitter) on individual data points.
        """
        jitter = np.random.uniform(-noise_level, noise_level, size=flux.shape)
        return flux + jitter

    @staticmethod
    def random_mask(flux, max_mask_frac=0.03, num_masks=2):
        """
        Masks small continuous segments of the sequence (replaced by the baseline/median flux)
        to simulate spacecraft gaps, data dropouts, or momentum dumps.
        """
        augmented = flux.copy()
        median_val = np.median(flux)
        seq_len = len(flux)
        
        for _ in range(num_masks):
            mask_len = int(np.random.uniform(0.005, max_mask_frac) * seq_len)
            if mask_len <= 0:
                continue
            start_idx = np.random.randint(0, seq_len - mask_len)
            augmented[start_idx:start_idx + mask_len] = median_val
            
        return augmented

    @staticmethod
    def temporal_stretch(flux, stretch_range=(0.95, 1.05)):
        """
        Stretches or compresses the time series slightly using linear interpolation.
        This simulates differences in transit duration/orbital periods.
        """
        factor = np.random.uniform(stretch_range[0], stretch_range[1])
        seq_len = len(flux)
        orig_indices = np.arange(seq_len)
        new_indices = np.arange(seq_len) * factor
        
        # Clip indices to prevent out of bounds and interpolate
        new_indices = np.clip(new_indices, 0, seq_len - 1)
        return np.interp(new_indices, orig_indices, flux)

    @classmethod
    def apply_augmentations(cls, flux, p=0.6):
        """
        Applies a pipeline of random time-series augmentations with a given probability.
        """
        if np.random.rand() > p:
            return flux
            
        augmented = flux.copy()
        
        # Apply select augmentations randomly to maintain natural variability
        if np.random.rand() < 0.4:
            augmented = cls.add_gaussian_noise(augmented)
        if np.random.rand() < 0.4:
            augmented = cls.random_shift(augmented)
        if np.random.rand() < 0.4:
            augmented = cls.random_scale(augmented)
        if np.random.rand() < 0.3:
            augmented = cls.flux_jitter(augmented)
        if np.random.rand() < 0.3:
            augmented = cls.random_mask(augmented)
        if np.random.rand() < 0.3:
            augmented = cls.temporal_stretch(augmented)
            
        return augmented
