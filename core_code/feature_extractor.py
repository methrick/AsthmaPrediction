import sys, os

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(current_file_path + '/../../')
sys.path.append(root_dir)  # To get the Root the directory

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from core_code.Feature_Extraction.AsthmaAnalyzer import AsthmaAnalyzer
import json
from collections import namedtuple
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

# Steps
'''
Steps :
    1. Get file path 
    2. Process the data and extract features
    3. Save it to DB     
'''

client = ''


def set_up_db():
    global client
    dotenv_path = root_dir + '/config/.env'
    # Load file from the path.
    load_dotenv(dotenv_path)
    client = MongoClient(os.getenv('DB_HOST'), int(os.getenv('DB_PORT')))

    # cred = credentials.Certificate(root_dir + '/config/serviceAccountKey.json')
    # firebase_admin.initialize_app(cred)


def get_breath_features_cb(audio_file_name, audio_file_path):
    asthma_analyzer = AsthmaAnalyzer('%s' % audio_file_path,
                                     '%s' % audio_file_name)
    power, energy, phase, ASE, Ti, number_of_rows, selected_data = asthma_analyzer.get_signal_features()
    return ASE, Ti, energy, number_of_rows, phase, power, selected_data


def save_each_sample(cb):
    is_training = False
   # data = "***REMOVED***\"label\":\"normal\",\"data_type\":\"training\",\"breath_files\":[***REMOVED***\"file_name\":\"normal-training-audio[0]-1526170690677.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[0]-1526170690677.wav\"***REMOVED***,***REMOVED***\"file_name\":\"normal-training-audio[1]-1526170690703.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[1]-1526170690703.wav\"***REMOVED***,***REMOVED***\"file_name\":\"normal-training-audio[2]-1526170690709.wav\",\"file_path\":\"/Users/JasemAl-sadi/WebstormProjects/asthma/samples/normal-training-audio[2]-1526170690709.wav\"***REMOVED***]***REMOVED***"
    if len(sys.argv) < 2:
        raise  Exception('No argument sent')
    data = sys.argv[1]
    data = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    if not hasattr(data, 'breath_files'):
        raise Exception('No Breath sound files path sent, it should be as array with name breath_files')

    if hasattr(data, 'label'):
        is_training = True

    all_records = data.breath_files
    for single_record in all_records:
        if not hasattr(single_record, 'file_name') or not hasattr(single_record, 'file_path'):
            raise Exception('Some files you send doesn"t has file_name or file_path attribute')
        audio_file_name = single_record.file_name
        audio_file_path = single_record.file_path
        # db = firestore.client()
        # doc_ref = db.collection('records').document('rec_collect')
        is_urgent_label = -1
        data_type = 'real_data'
        if is_training:
            is_urgent_label = data.label
            data_type = data.data_type

            ASE, Ti, energy, number_of_rows, phase, power, selected_data = cb(audio_file_name, audio_file_path)
            db = client[os.getenv('DB_DATABASE')]
            current_record = ***REMOVED***
                'is_urgent': -1,
                'is_urgent_label': is_urgent_label,
                'type_of_record': data_type,
                'wav_file_path': audio_file_path,
                'features': ***REMOVED***
                    'ASE': json.dumps(ASE.tolist()),
                    'Ti': json.dumps(Ti.tolist()),
                    'Energy': json.dumps(energy.tolist()),
                    'Phase': json.dumps(phase.tolist()),
                    'Power': json.dumps(power.tolist()),
                ***REMOVED***,
                'created_at': time.time(),
            ***REMOVED***
            current_record_id = db.records.insert_one(current_record).inserted_id


start = time.time()
print("\n Start Time  = " + str(start) + "\n")
set_up_db()
save_each_sample(get_breath_features_cb)
End = time.time()
elapsed = End - start
print("End Time = "+str(End) + "\n")
print("Difference = " + str(elapsed))
