class ManyFacesException(Exception):
    status_code = 400

    def __init__(self, message, status_code=None):
        super().__init__()
        self.__message = message
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict(())
        rv['message'] = self.__message
        return rv
