import logging
def init_logging(log_path):
  log = logging.getLogger()
  log.setLevel(logging.INFO)
  logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

  fileHandler = logging.FileHandler(log_path)
  fileHandler.setFormatter(logFormatter)
  log.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  log.addHandler(consoleHandler)
