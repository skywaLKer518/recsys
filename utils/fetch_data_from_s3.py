import time
import datetime
import logging
import os.path
import pickle
import sys
import pandas as pd
from io import BytesIO
from s3_utility import Connect2S3


def ReadFromS3(mys3bucket, file_prefix, logger, filename=None):
    """ Read data file off of specified S3 bucket.
    If filename is None, then download the file following the "file_prefix"
    with the latest timestamp. Else, download the file specified by "filename"

    Args:
            mys3bucket <str>: name of the AWS S3 bucket to access
            file_prefix <str>: source directory + file prefix used to filter
            logger <logging obj>: for logging
            filename <str>: name of file to read (default is None)

    Returns:
            data <bytes>: raw bytes data from the S3 source data file
    """
    # start timer!
    start = time.time()

    # connect to source s3 bucket
    bucket = Connect2S3(mys3bucket).Bucket(mys3bucket)

    # if filename is not specified
    if filename is None:

        # retrieve max timestamp on all obj following "file_prefix"
        timestamps = []

        # iterate over all objects within the bucket satisfying the prefix
        for obj in bucket.objects.filter(Prefix=file_prefix):
            try:
                # collect the timestamps
                ts = datetime.datetime.strptime(
                    str(obj.key)[-23:-4], '%Y-%m-%d-%H-%M-%S')
                timestamps.append(ts)
            except:
                pass

        # error if no objects within the bucket satify the prefix filter criteria
        try:
            # get max timestamp
            max_timestamp = datetime.datetime.strftime(
                max(timestamps), '%Y-%m-%d-%H-%M-%S')
        except:
            logger("no file with prefix \"%s\" was found." %
                   (file_prefix))
            sys.exit()

        # try loading the obj following "file_prefix" + "max_timestamp" + ".csv" name
        for obj in bucket.objects.filter(Prefix=file_prefix + max_timestamp + '.csv'):
            # read source file
            data = obj.get()['Body'].read()
            logger("reading file: " + file_prefix +
                   max_timestamp + '.csv')
            logger("read file took %.2f seconds." % (time.time() - start))
        try:
            return data
        except:
            logger("no file: \"%s\" was found." %
                   (file_prefix + max_timestamp + '.csv'))
            sys.exit()

    # if filename is specified
    else:
        # find source file obj within the s3 bucket using the specific filename
        # for filter criteria
        for obj in bucket.objects.filter(Prefix=filename):
            # read source file
            data = obj.get()['Body'].read()
            logger("reading file: %s" % filename)
            logger("read file took %d seconds." % (time.time() - start))

        try:
            return data
        except:
            logger.error("target file \"%s\" does not exist." % (filename))
            sys.exit()


def fetch_data_from_s3(mys3bucket, file_prefix, filename, logger):
    """ Intermediary wrapper function that calls "ReadFromS3" to read static
    data file off of AWS S3. Converts the raw bytes into dataframe obj and
    perform formating (rename columsn & typecasting) to downstream processes.

    Args:
            mys3bucket <str>: name of AWS S3 bucket to access
            file_prefix <str>: file-prefix to perform filter to locate the
                               desired data file in the S3 bucket
            filename <str>: name of data file in the AWS S3 bucket
            column_names <list>: list of strings specifying data column names
            logger <logging obj>: for logging

    Return:
            df <pandas.DataFrame>: processed dataframe obj containing source data
    """

    start = time.time()

    # read source data off of s3 bucket
    data = ReadFromS3(mys3bucket, file_prefix, logger, filename)

    try:
        # convert bytes data into pandas dataframe
        df = pd.read_csv(BytesIO(data), sep="\t", header=0)
        logger("load data & convert to dataframe took %.2f seconds" %
               (time.time() - start))
        return df
    except:
        logger('failed converting raw bytes to dataframe')
        logger(sys.exc_info())
        sys.exit()
