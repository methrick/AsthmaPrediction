from core_code.AsthmaAnalyzer import AsthmaAnalyzer

audio_file_path = '../samples/wheeze_with_crackles.wav'
audio_file_name = 'wheeze_with_crackles'

asthma_analyzer = AsthmaAnalyzer('%s' % audio_file_path,
                                 '%s' % audio_file_name)
asthma_analyzer.get_signal_features()
