conf_wavelet_standard_lr = {'model_name': 'Wavelet+LR', 'model_type': 'WAVELET',
                            'parameters': dict(
                                regularizer_C=.001,
                                classifier='LR'
                            )}

conf_wavelet_standard_rf = {'model_name': 'Wavelet+RF', 'model_type': 'WAVELET',
                            'parameters': dict(
                                regularizer_C=.001,
                                classifier='RF'
                            )}

conf_wavelet_standard_nn = {'model_name': 'Wavelet+NN', 'model_type': 'WAVELET',
                            'parameters': dict(
                                regularizer_C=.001,
                                classifier='NN'
                            )}