def calculate_entropy(list_values):
        counter_values = Counter(list_values).most_common()
        probabilities = [elem[1] / len(list_values) for elem in counter_values]
        entropy = scipy.stats.entropy(probabilities)
        return entropy

def calculate_statistics(list_values):
        n5 = np.nanpercentile(list_values, 5)
        n25 = np.nanpercentile(list_values, 25)
        n75 = np.nanpercentile(list_values, 75)
        n95 = np.nanpercentile(list_values, 95)
        median = np.nanpercentile(list_values, 50)
        mean = np.nanmean(list_values)
        std = np.nanstd(list_values)
        var = np.nanvar(list_values)
        rms = np.nanmean(np.sqrt(list_values ** 2))
        return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
        zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
        entropy = wavelet.calculate_entropy(list_values)
        crossings = wavelet.calculate_crossings(list_values)
        statistics = wavelet.calculate_statistics(list_values)
        return [entropy] + crossings + statistics

def get_single_ecg_features(signal, waveletname='db6'):
        features = []
        for channel in signal.T:
            list_coeff = pywt.wavedec(channel, wavelet=waveletname, level=5)
            channel_features = []
            for coeff in list_coeff:
                channel_features += wavelet.get_features(coeff)
            features.append(channel_features)
        return np.array(features).flatten()

def get_ecg_features(ecg_data, parallel=True):
    if parallel:
        pool = multiprocessing.Pool(18)
        return np.array(pool.map(wavelet.get_single_ecg_features, ecg_data))
    else:
        list_features = []
        for signal in tqdm(ecg_data):
            features = wavelet.get_single_ecg_features(signal)
            list_features.append(features)
        return np.array(list_features)

# for keras models
# def keras_macro_auroc(y_true, y_pred):
#    return tf.py_func(macro_auroc, (y_true, y_pred), tf.double)