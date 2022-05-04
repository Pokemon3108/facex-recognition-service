class DuplicateUsernameException(Exception):
    status_code = 400

    def __init__(self, message=None, status_code=None):
        super().__init__()
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self):
        rv = dict(())
        rv['message'] = self.message
        return rv
