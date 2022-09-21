def __init__(self, name, n_classes, freq, outputfolder, input_shape, regularizer_C=.001, classifier='RF'):
            # Disclaimer: This model assumes equal shapes across all samples!
            # standard parameters
            self.name = name
            self.outputfolder = outputfolder
            self.n_classes = n_classes
            self.freq = freq
            self.regularizer_C = regularizer_C
            self.classifier = classifier
            self.dropout = .25
            self.activation = 'relu'
            self.final_activation = 'sigmoid'
            self.n_dense_dim = 128
            self.epochs = 30

def fit(self, X_train, y_train, X_val, y_val):
            XF_train = wavelet.get_ecg_features(X_train)
            XF_val = wavelet.get_ecg_features(X_val)

            if self.classifier == 'LR':
                if self.n_classes > 1:
                    clf = OneVsRestClassifier(
                        LogisticRegression(C=self.regularizer_C, solver='lbfgs', max_iter=1000, n_jobs=-1))
                else:
                    clf = LogisticRegression(C=self.regularizer_C, solver='lbfgs', max_iter=1000, n_jobs=-1)
                clf.fit(XF_train, y_train)
                pickle.dump(clf, open(self.outputfolder + 'clf.pkl', 'wb'))
            elif self.classifier == 'RF':
                clf = RandomForestClassifier(n_estimators=1000, n_jobs=16)
                clf.fit(XF_train, y_train)
                pickle.dump(clf, open(self.outputfolder + 'clf.pkl', 'wb'))
            elif self.classifier == 'NN':
                # standardize input data
                ss = StandardScaler()
                XFT_train = ss.fit_transform(XF_train)
                XFT_val = ss.transform(XF_val)
                pickle.dump(ss, open(self.outputfolder + 'ss.pkl', 'wb'))
                # classification stage
                input_x = Input(shape=(XFT_train.shape[1],))
                x = Dense(self.n_dense_dim, activation=self.activation)(input_x)
                x = Dropout(self.dropout)(x)
                y = Dense(self.n_classes, activation=self.final_activation)(x)
                self.model = Model(input_x, y)

                self.model.compile(optimizer='adamax', loss='binary_crossentropy')  # , metrics=[keras_macro_auroc])
                # monitor validation error
                mc_loss = ModelCheckpoint(self.outputfolder + 'best_loss_model.h5', monitor='val_loss', mode='min',
                                          verbose=1, save_best_only=True)
                # mc_score = ModelCheckpoint(self.outputfolder +'best_score_model.h5', monitor='val_keras_macro_auroc', mode='max', verbose=1, save_best_only=True)
                self.model.fit(XFT_train, y_train, validation_data=(XFT_val, y_val), epochs=self.epochs, batch_size=128,
                               callbacks=[mc_loss])  # , mc_score])
                self.model.save(self.outputfolder + 'last_model.h5')

def predict(self, X):
            XF = wavelet.get_ecg_features(X)
            if self.classifier == 'LR':
                clf = pickle.load(open(self.outputfolder + 'clf.pkl', 'rb'))
                if self.n_classes > 1:
                    return clf.predict_proba(XF)
                else:
                    return clf.predict_proba(XF)[:, 1][:, np.newaxis]
            elif self.classifier == 'RF':
                clf = pickle.load(open(self.outputfolder + 'clf.pkl', 'rb'))
                y_pred = clf.predict_proba(XF)
                if self.n_classes > 1:
                    return np.array([yi[:, 1] for yi in y_pred]).T
                else:
                    return y_pred[:, 1][:, np.newaxis]
            elif self.classifier == 'NN':
                ss = pickle.load(open(self.outputfolder + 'ss.pkl', 'rb'))  #
                XFT = ss.transform(XF)
                model = load_model(self.outputfolder + 'best_loss_model.h5')  # 'best_score_model.h5', custom_objects={'keras_macro_auroc': keras_macro_auroc})
                return model.predict(XFT)