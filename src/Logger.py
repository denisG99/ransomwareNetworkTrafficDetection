class Logger:
    def __init__(self, failure, success, path):
        self.__log_fail = failure
        self.__log_success = success
        self.__log_path = path

    def log_writer(self, file, res):
        with open(self.__log_path, 'a') as log:
            if res == 0:
                log.write(f"{file} {self.__log_success}\n")
            else:
                log.write(f"{file} {self.__log_fail}\n")
