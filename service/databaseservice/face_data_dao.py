from service.databaseservice.connector import MongoConnector

connector = MongoConnector(db_name='facex',
                           collection_name='faceData',
                           hostname="localhost",
                           port=27017)


def save_known_face(dict):
    connector.save(dict)

