from flask import jsonify

from app import app
from exception.not_found_exception import NotFoundException


@app.errorhandler(NotFoundException)
def not_found_resource(e):
    return jsonify(e.to_dict()), e.status_code
