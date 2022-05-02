from flask import jsonify

from app import app
from exception.DuplicateUsernameException import DuplicateUsernameException
from exception.ManyFacesException import ManyFacesException
from exception.NotFoundException import NotFoundException


@app.errorhandler(NotFoundException)
def process_not_found_resource(e):
    return jsonify(e.to_dict()), e.status_code

@app.errorhandler(ManyFacesException)
def process_many_faces_exception(e):
    return jsonify(e.to_dict()), e.status_code


@app.errorhandler(DuplicateUsernameException)
def process_duplicate_username_exception(e):
    return jsonify(e.to_dict()), e.status_code
