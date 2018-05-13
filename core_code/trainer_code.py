import sys, os

current_file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(current_file_path + '/../../')
sys.path.append(root_dir)  # To get the Root the directory

from core_code.Feature_Extraction.AsthmaAnalyzer import AsthmaAnalyzer
import json
from collections import namedtuple
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv


def get_datasets():
    dotenv_path = root_dir + '/config/.env'
    # Load file from the path.
    load_dotenv(dotenv_path)
    client = MongoClient(os.getenv('DB_HOST'), int(os.getenv('DB_PORT')))
    db = client[os.getenv('DB_DATABASE')]
    all_records = db.records.find(***REMOVED******REMOVED***)
    labels = []
    breath_features = []
    for single_record in all_records:
        single_label = single_record.get('is_urgent_label')
        labels.append(single_label)
        single_breath_features = single_record.get('features')
        breath_features.append(single_breath_features)
    return breath_features, labels


def train_svm(features, labels):
    pass


def train_knn(features, labels):
    pass


def train_ann(features, labels):
    pass


features, labels = get_datasets()

train_svm(features, labels)
train_knn(features, labels)
train_ann(features, labels)
