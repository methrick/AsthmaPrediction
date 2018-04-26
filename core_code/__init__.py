from core_code.AsthmaAnalyzer import AsthmaAnalyzer

asthma_analyzer = AsthmaAnalyzer('../samples/NormalBreathSound.wav',
                                 'Normal Breath Sound')
asthma_analyzer.get_signal_features()
