from flask import jsonify

from app import app
from exception.many_faces_exception import ManyFacesException
from exception.not_found_exception import NotFoundException


@app.errorhandler(NotFoundException)
def not_found_resource(e):
    print(type(e))
    return jsonify(e.to_dict()), e.status_code

@app.errorhandler(ManyFacesException)
def many_faces_exception(e):
    return jsonify(e.to_dict()), e.status_code
