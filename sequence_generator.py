import numpy as np
import random

np.random.seed(1)
random.seed(1)

class Single_Signal_Generator:

    def __init__(self, total_timesteps, period_range, 
                    amplitude_range, noise_amplitude_ratio, sample_type="multi_sin_concat_with_base_whp",
                    base_period_ratio=(2, 4), base_amplitude_range=(20, 80)):

        self.total_timesteps = total_timesteps
        self.noise_amplitude_ratio = noise_amplitude_ratio
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.base_period_range = (base_period_ratio[0]*period_range[1], base_period_ratio[1]*period_range[1])
        self.base_amplitude_range = base_amplitude_range
        self.base = False
        self.half_period = False
        self.loaded = False
        self.loaded_signals = None
        self.no_of_loaded_signals = None
        self.counter = 0

    def build_signals(self, filename, no_of_samples, sample_type="multi_sin_concat_with_base_whp", full_episode=True):
        if self.loaded:
            print("Signals already loaded!")
            return None

        signals = np.zeros((self.total_timesteps, no_of_samples))
        for i in range(no_of_samples):
            signals[:, i] = self.sample().reshape((signals.shape[0]))

        np.save(filename, signals)

    def load(self, filename):
        self.loaded_signals = np.load(filename)
        self.loaded = True
        self.no_of_loaded_signals = self.loaded_signals.shape[1]

    def sample(self, sample_type="multi_sin_concat_with_base_whp", full_episode=True):

        if self.loaded:
            if self.counter == self.no_of_loaded_signals:
                print("All loaded signals returned. Starting from the first signal again.")
                self.counter = 0

            return self.loaded_signals[:, self.counter].reshape((self.total_timesteps, 1))
        else:
            if sample_type == "multi_sin_concat_with_base_whp":
                self.base = True
                self.half_period = True
                return self._sample_multi_sin()
            elif sample_type == "single_sin":
                self.base = False
                self.half_period = False
                return self._sample_single_sin()
            elif sample_type == "multi_sin_concat":
                return self._sample_multi_sin()
            elif sample_type == "multi_sin_concat_whp":
                self.half_period = True
                return self._sample_multi_sin()
            else:
                print("Cannot recognise type. Defaulting to 'multi_sin_concat_with_base_whp'.")
                self.half_period = True
                return self._sample_multi_sin()

        
    def _random_sin(self, base=False, full_episode=False):

        if base:
            period = random.randrange(self.base_period_range[0], self.base_period_range[1])
            amplitude = random.randrange(self.base_amplitude_range[0], self.base_amplitude_range[1])
            noise = 0
        else:
            period = random.randrange(self.period_range[0], self.period_range[1])
            amplitude = random.randrange(self.amplitude_range[0], self.amplitude_range[1])
            noise = self.noise_amplitude_ratio * amplitude

        if full_episode:
            length = self.total_timesteps
        else:
            if self.half_period:
                length = int(random.randrange(1,4) * 0.5 * period)
            else:
                length = period

        signal_value = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
        signal_value += np.random.random(signal_value.shape) * noise

        return signal_value

    def _sample_single_sin(self):
        sample_container = []

        sample, function_name = self._random_sin(full_episode=True)

        sample_container.append(sample)

        return np.array(sample_container).T

    def _sample_multi_sin(self):
        sample_container = []
        sample = []
        while True:
            sample = np.append(sample, self._random_sin(full_episode=False)[0])
            if len(sample) > self.total_timesteps:
                break

        if self.base:
            base = self._random_sin(base=True, full_episode=True)
            sample_container.append(sample[:self.total_timesteps] + base)
            return np.array(sample_container).T
        else:
            sample_container.append(sample[:self.total_timesteps])
            return np.array(sample_container).T

if __name__ == '__main__':
    gen = Single_Signal_Generator(180, (10, 40), (5, 80), 0.5)

    filename = "Generated Signals.npy"
    gen.build_signals(filename, 1000)
    gen.load(filename)
    print(gen.sample())
    print(gen.loaded_signals.shape)
