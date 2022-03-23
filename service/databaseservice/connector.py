from pymongo import MongoClient


class MongoConnector:

    def __init__(self, db_name, collection_name, hostname='localhost', port=27017, username="alexbw", password=""):
        client = MongoClient(hostname, port)
        db = client[db_name]
        self.collection = db[collection_name]

    def save(self, obj_to_save):
        self.collection.insert_one(obj_to_save)

