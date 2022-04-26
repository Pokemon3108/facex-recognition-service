class DuplicateUsernameException(Exception):
    __status_code = 400

    def __init__(self, message=None, status_code=None):
        super().__init__()
        if message is not None:
            self.__message = message
        if status_code is not None:
            self.__status_code = status_code

    def to_dict(self):
        rv = dict(())
        rv['message'] = self.__message
        return rv

    def get_status_code(self):
        return self.__status_code

    def get_message(self):
        return self.__message
