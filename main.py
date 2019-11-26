import os
import sys
import logging
import time
import datetime
import psutil
from logging import FileHandler

sys.path.insert(0, './utils')
sys.path.insert(0, './hmf')
sys.path.insert(0, './lstm')
sys.path.insert(0, './attributes')

from hmf_class import hmf
from lstm_class import lstm
from load_config import config_ENV_setup
import tensorflow as tf


def main(_):
        # create directories if they don't exist already
    # ----------------------------------------------
    if not os.path.exists('log'):
        os.makedirs('log')

    # setup logging, create log file handler & formatter
    # ----------------------------------------------
    logFileHandler = FileHandler(
        filename='log/rec_pipeline_%s.log' % (time.strftime('%Y-%m-%d_%H-%M-%S')))
    logFormatter = logging.Formatter(
        "'%(asctime)s - %(name)s - %(levelname)s - %(message)s'")
    # create and configure logger obj
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    logFileHandler.setFormatter(logFormatter)
    logger.addHandler(logFileHandler)

    # log initial CPU and RAM status
    logger.info('starting main process')
    logger.info('cpu at %.2f%%' % (psutil.cpu_percent()))
    logger.info('memory state: %s' % str(psutil.virtual_memory()))

    # load config files
    config_file_path_dict = {'user_item_recommender_config': 'config/user_item_recommender_config.json'
                             }
    configs = config_ENV_setup(config_file_path_dict)

    # build recommendations

    if configs['user_item_recommender_config']['model'] == "hmf":
        hmf_model = hmf(configs['user_item_recommender_config'])

        if configs['user_item_recommender_config']['recommend']:
            hmf_model.compute_scores()
        else:
            hmf_model.train()
    else:
        lstm_model = lstm(configs['user_item_recommender_config'])

        if configs['user_item_recommender_config']['beam_search']:
            lstm_model.beam_search()

        if configs['user_item_recommender_config']['ensemble']:
            lstm_model.ensemble()

        if configs['user_item_recommender_config']['recommend']:
            lstm_model.recommend()
        else:
            lstm_model.train()


if __name__ == "__main__":
    tf.app.run()
