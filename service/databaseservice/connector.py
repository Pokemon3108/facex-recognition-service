from pymongo import MongoClient


class MongoConnector:

    def __init__(self, db_name, collection_name, hostname='localhost', port=27017):
        client = MongoClient(hostname, port)
        db = client[db_name]
        self.__collection = db[collection_name]

    def save(self, doc_to_save):
        self.__collection.insert_one(doc_to_save)

    def search(self, search_criteria):
        return self.__collection.find_one(search_criteria)

    def read_all(self):
        return self.__collection.find({})

    def update(self, filter_criteria, update_criteria):
        return self.__collection.find_one_and_update(filter_criteria, update_criteria)

