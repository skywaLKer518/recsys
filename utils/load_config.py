import os
import sys
import json
import logging


def load_configurations(config_filepath):
    """ wrapper function for setting up environment and configuration
    related information.

    Args:
            config_filepath <str>: local filepath to the config file
    Returns:
            ENV <str>: specifies which environment to work in
            config <json>: holds config info
    """

    logger = logging.getLogger('main.load_config.load_configurations')

    # load configuration files from local filepath
    logger.info('loading %s' % config_filepath)
    # get data source & model configuration settings
    try:
        with open(config_filepath) as f:
            config = json.load(f)
    except:
        logger.info('error load config: %s' % config_filepath)
        logger.error(sys.exc_info())
        sys.exit()

    return config


def get_ENV():
    """ reads and check the ENV parameter for validity.

    Args: None
    Returns:
            ENV <str>:  specifies the envrionment the script is to run on;
                                    valid choices are "local", "development", "qa",
                                    "staging", and "production".
    """

    logger = logging.getLogger('main.load_config.get_ENV')

    # loads the environment variable that is passed to the shell script.
    try:
        ENV = os.environ['ENV'].lower()
    except:
        logger.info('no ENV value specified, default to \"local\"')
        ENV = 'local'
        return ENV

    if ENV in ['local', 'development', 'qa', 'staging', 'production']:
        logger.info('using environment: %s' % ENV)
        return ENV
    else:
        logger.error('invalid ENV specified')
        sys.exit()


def config_ENV_setup(config_file_path_dict):
    """ wrapper script to handle loading of necessary config files and
    environments for the main program

    Args:
            config_file_path_dict <dict>: dictionary where each key is the
                                          name of the config file,
                                          values are the local filepath for
                                          the config file.
    Returns:
            config_dict <dict>: dictionary where each key is the name of
                                the config file, values are the json object
                                associated with the config. Also provides an
                                addition "ENV" key that specifies the
                                environment parameter.
    """

    config_dict = {}

    for config_name in config_file_path_dict.keys():

        config_filepath = config_file_path_dict[config_name]
        config_dict[config_name] = load_configurations(config_filepath)
        config_dict[config_name]['environment'] = get_ENV()

    return config_dict
