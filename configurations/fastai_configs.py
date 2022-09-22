conf_fastai_resnet1d18 = {'model_name': 'fastai_resnet1d18', 'model_type': 'FastaiModel',
                          'parameters': dict()}

conf_fastai_resnet1d34 = {'model_name': 'fastai_resnet1d34', 'model_type': 'FastaiModel',
                          'parameters': dict()}

conf_fastai_resnet1d50 = {'model_name': 'fastai_resnet1d50', 'model_type': 'FastaiModel',
                          'parameters': dict()}

conf_fastai_resnet1d101 = {'model_name': 'fastai_resnet1d101', 'model_type': 'FastaiModel',
                           'parameters': dict()}

conf_fastai_resnet1d152 = {'model_name': 'fastai_resnet1d152', 'model_type': 'FastaiModel',
                           'parameters': dict()}

conf_fastai_resnet1d_wang = {'model_name': 'fastai_resnet1d_wang', 'model_type': 'FastaiModel',
                             'parameters': dict()}

conf_fastai_wrn1d_22 = {'model_name': 'fastai_wrn1d_22', 'model_type': 'FastaiModel',
                        'parameters': dict()}

conf_fastai_xresnet1d18 = {'model_name': 'fastai_xresnet1d18', 'model_type': 'FastaiModel',
                           'parameters': dict()}

conf_fastai_xresnet1d34 = {'model_name': 'fastai_xresnet1d34', 'model_type': 'FastaiModel',
                           'parameters': dict()}

conf_fastai_xresnet1d50 = {'model_name': 'fastai_xresnet1d50', 'model_type': 'FastaiModel',
                           'parameters': dict()}

# more xresnet50s
conf_fastai_xresnet1d50_ep30 = {'model_name': 'fastai_xresnet1d50_ep30', 'model_type': 'FastaiModel',
                                'parameters': dict(epochs=30)}

conf_fastai_xresnet1d50_validloss_ep30 = {'model_name': 'fastai_xresnet1d50_validloss_ep30',
                                          'model_type': 'FastaiModel',
                                          'parameters': dict(early_stopping="valid_loss", epochs=30)}

conf_fastai_xresnet1d50_macroauc_ep30 = {'model_name': 'fastai_xresnet1d50_macroauc_ep30', 'model_type': 'FastaiModel',
                                         'parameters': dict(early_stopping="macro_auc", epochs=30)}

conf_fastai_xresnet1d50_fmax_ep30 = {'model_name': 'fastai_xresnet1d50_fmax_ep30', 'model_type': 'FastaiModel',
                                     'parameters': dict(early_stopping="fmax", epochs=30)}

conf_fastai_xresnet1d50_ep50 = {'model_name': 'fastai_xresnet1d50_ep50', 'model_type': 'FastaiModel',
                                'parameters': dict(epochs=50)}

conf_fastai_xresnet1d50_validloss_ep50 = {'model_name': 'fastai_xresnet1d50_validloss_ep50',
                                          'model_type': 'FastaiModel',
                                          'parameters': dict(early_stopping="valid_loss", epochs=50)}

conf_fastai_xresnet1d50_macroauc_ep50 = {'model_name': 'fastai_xresnet1d50_macroauc_ep50', 'model_type': 'FastaiModel',
                                         'parameters': dict(early_stopping="macro_auc", epochs=50)}

conf_fastai_xresnet1d50_fmax_ep50 = {'model_name': 'fastai_xresnet1d50_fmax_ep50', 'model_type': 'FastaiModel',
                                     'parameters': dict(early_stopping="fmax", epochs=50)}

conf_fastai_xresnet1d101 = {'model_name': 'fastai_xresnet1d101', 'model_type': 'FastaiModel',
                            'parameters': dict()}

conf_fastai_xresnet1d152 = {'model_name': 'fastai_xresnet1d152', 'model_type': 'FastaiModel',
                            'parameters': dict()}

conf_fastai_xresnet1d18_deep = {'model_name': 'fastai_xresnet1d18_deep', 'model_type': 'FastaiModel',
                                'parameters': dict()}

conf_fastai_xresnet1d34_deep = {'model_name': 'fastai_xresnet1d34_deep', 'model_type': 'FastaiModel',
                                'parameters': dict()}

conf_fastai_xresnet1d50_deep = {'model_name': 'fastai_xresnet1d50_deep', 'model_type': 'FastaiModel',
                                'parameters': dict()}

conf_fastai_xresnet1d18_deeper = {'model_name': 'fastai_xresnet1d18_deeper', 'model_type': 'FastaiModel',
                                  'parameters': dict()}

conf_fastai_xresnet1d34_deeper = {'model_name': 'fastai_xresnet1d34_deeper', 'model_type': 'FastaiModel',
                                  'parameters': dict()}

conf_fastai_xresnet1d50_deeper = {'model_name': 'fastai_xresnet1d50_deeper', 'model_type': 'FastaiModel',
                                  'parameters': dict()}

conf_fastai_inception1d = {'model_name': 'fastai_inception1d', 'model_type': 'FastaiModel',
                           'parameters': dict()}

conf_fastai_inception1d_input256 = {'model_name': 'fastai_inception1d_input256', 'model_type': 'FastaiModel',
                                    'parameters': dict(input_size=256)}

conf_fastai_inception1d_input512 = {'model_name': 'fastai_inception1d_input512', 'model_type': 'FastaiModel',
                                    'parameters': dict(input_size=512)}

conf_fastai_inception1d_input1000 = {'model_name': 'fastai_inception1d_input1000', 'model_type': 'FastaiModel',
                                     'parameters': dict(input_size=1000)}

conf_fastai_inception1d_no_residual = {'model_name': 'fastai_inception1d_no_residual', 'model_type': 'FastaiModel',
                                       'parameters': dict()}

conf_fastai_fcn = {'model_name': 'fastai_fcn', 'model_type': 'FastaiModel',
                   'parameters': dict()}

conf_fastai_fcn_wang = {'model_name': 'fastai_fcn_wang', 'model_type': 'FastaiModel',
                        'parameters': dict()}

conf_fastai_schirrmeister = {'model_name': 'fastai_schirrmeister', 'model_type': 'FastaiModel',
                             'parameters': dict()}

conf_fastai_sen = {'model_name': 'fastai_sen', 'model_type': 'FastaiModel',
                   'parameters': dict()}

conf_fastai_basic1d = {'model_name': 'fastai_basic1d', 'model_type': 'FastaiModel',
                       'parameters': dict()}

conf_fastai_lstm = {'model_name': 'fastai_lstm', 'model_type': 'FastaiModel',
                    'parameters': dict(lr=1e-3)}

conf_fastai_gru = {'model_name': 'fastai_gru', 'model_type': 'FastaiModel',
                   'parameters': dict(lr=1e-3)}

conf_fastai_lstm_bidir = {'model_name': 'fastai_lstm_bidir', 'model_type': 'FastaiModel',
                          'parameters': dict(lr=1e-3)}

conf_fastai_gru_bidir = {'model_name': 'fastai_gru', 'model_type': 'FastaiModel',
                         'parameters': dict(lr=1e-3)}

conf_fastai_lstm_input1000 = {'model_name': 'fastai_lstm_input1000', 'model_type': 'FastaiModel',
                              'parameters': dict(input_size=1000, lr=1e-3)}

conf_fastai_gru_input1000 = {'model_name': 'fastai_gru_input1000', 'model_type': 'FastaiModel',
                             'parameters': dict(input_size=1000, lr=1e-3)}

conf_fastai_schirrmeister_input500 = {'model_name': 'fastai_schirrmeister_input500', 'model_type': 'FastaiModel',
                                      'parameters': dict(input_size=500)}

conf_fastai_inception1d_input500 = {'model_name': 'fastai_inception1d_input500', 'model_type': 'FastaiModel',
                                    'parameters': dict(input_size=500)}

conf_fastai_fcn_wang_input500 = {'model_name': 'fastai_fcn_wang_input500', 'model_type': 'FastaiModel',
                                 'parameters': dict(input_size=500)}