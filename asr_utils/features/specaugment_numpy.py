import numpy as np


class FreqMasking:
    def __init__(self, num_masks: int = 1, mask_factor: float = 27):
        self.num_masks = num_masks
        self.mask_factor = mask_factor

    def augment(self, spectrogram):
        """
        Masking the frequency channels (shape[1])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        T, F, V = spectrogram.shape
        for _ in range(self.num_masks):
            f = int(np.random.uniform(low=0, high=self.mask_factor))
            f = min([f, F])
            f0 = int(np.random.uniform(low=0, high=(F - f)))
            mask = np.concatenate(
                [
                    np.ones([T, f0, V], dtype=spectrogram.dtype),
                    np.zeros([T, f, V], dtype=spectrogram.dtype),
                    np.ones([T, F - f0 - f, V], dtype=spectrogram.dtype),
                ],
                axis=1,
            )
            spectrogram = spectrogram * mask
        return spectrogram


class TimeMasking:
    def __init__(self, num_masks: int = 1, mask_factor: float = 100, p_upperbound: float = 1.0):
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound

    def augment(self, spectrogram):
        """
        Masking the time channel (shape[0])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        T, F, V = spectrogram.shape
        for _ in range(self.num_masks):
            t = int(np.random.uniform(low=0, high=self.mask_factor))
            t = np.min([t, int(T * self.p_upperbound)])
            t0 = int(np.random.uniform(low=0, high=(T - t)))
            mask = np.concatenate(
                [
                    np.ones([t0, F, V], dtype=spectrogram.dtype),
                    np.zeros([t, F, V], dtype=spectrogram.dtype),
                    np.ones([T - t0 - t, F, V], dtype=spectrogram.dtype),
                ],
                axis=0,
            )
            spectrogram = spectrogram * mask
        return spectrogram


class Augmentation:
    def __init__(self, config):
        self.prob = float(config.get('prob', 0.5))
        self.augmentations = []
        freq_masking_config = config.get('freq_masking')
        time_masking_config = config.get('time_masking')
        self.augmentations.append(FreqMasking(**freq_masking_config))
        self.augmentations.append(TimeMasking(**time_masking_config))

    def augment(self, inputs):
        outputs = inputs
        for au in self.augmentations:
            p = np.random.uniform()
            outputs = np.where(np.less(p, self.prob), au.augment(outputs), outputs)
        return outputs


if __name__ == '__main__':
    augmentation = Augmentation({"time_masking": {"num_masks": 10,
                                                  "mask_factor": 100,
                                                  "p_upperbound": 0.05},
                                 "freq_masking": {"num_masks": 1,
                                                  "mask_factor": 27},
                                 'prob': 0.5})
    print(augmentation.augment(np.ones((5,5,1))))