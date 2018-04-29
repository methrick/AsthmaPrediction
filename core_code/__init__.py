from core_code.AsthmaAnalyzer import AsthmaAnalyzer

audio_file_path = '../samples/NormalBreathSound.wav'
audio_file_name = 'Normal Breath Sound'

asthma_analyzer = AsthmaAnalyzer('%s' % audio_file_path,
                                 '%s' % audio_file_name)
asthma_analyzer.get_signal_features()
