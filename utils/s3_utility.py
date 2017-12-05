import boto3
import botocore
import time
import datetime


def Connect2S3(mys3bucket):
	""" Checks the existence for AWS S3 bucket, if exists, 
	connect to S3. Returns the s3 connections obj. 
	If bucket does not exist, return None
	Note: system should have AWS CLI installed and credential 
	set up properly through AWS CLI's "aws configuration" command.
	See: http://boto3.readthedocs.io/en/latest/guide/quickstart.html#guide-quickstart
	
	Args:
	mys3bucket: name for the S3 bucket to connect to

	Returns:
	boto3.s3 connections obj
	"""

	# instantiate s3 obj
	s3 = boto3.resource('s3')
	# attempt to locate mys3bucket
	bucket = s3.Bucket(mys3bucket)
	try:
		s3.meta.client.head_bucket(Bucket=mys3bucket)
		return s3
	except botocore.exceptions.ClientError as e:
		# check if error is a 404 error, indicating bucket nonexistence
		if e.response['Error']['Code']=='404':
			print("[Error] Bucket \"%s\" does not exist." %mys3bucket)
		return None


def DownloadFromS3(mys3bucket, src_dir, dest_dir, filename):
	""" download specified file from specified AWS S3 bucket
	and directory.

	Args: 
	mys3bucket: name of the AWS S3 bucket to access
	src_dir: source directory location in the AWS S3 bucket
	dest_dir: destination directory to download file to
	filename: name of file to pull

	Returns:
	status: boolean value for success(True)/fail(False)
	"""

	# establish connection to s3 bucket
	s3 = Connect2S3(mys3bucket)
	# attempt to download file
	try:
		s3.meta.client.download_file(Bucket=mys3bucket,
		                             Key=src_dir+filename,
		                             Filename=dest_dir+filename)
		return True
	except AttributeError:
		print("Could not connecto S3 bucket, check if bucket \"%s\" exists"%mys3bucket)
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code']=='404':
			print("[Error] Source \"%s\" or destination \"%s\" does not exist."%(src_dir+filename, dest_dir+filename))
		return False



def UploadToS3(mys3bucket, src_dir, dest_dir, filename):
	""" push specified dir/file up onto specified S3 bucket.

	Args: 
	mys3bucket: name of the AWS S3 bucket to access
	src_dir: source directory location in the AWS S3 bucket
	dest_dir: destination directory to download file to
	filename: name of file to push

	Returns:
	status: boolean value for success(True)/fail(False)
	"""
	# establish connection to s3 bucket
	s3 = Connect2S3(mys3bucket)
	# attempt to push file
	try:
		data = open(src_dir+filename, 'rb')
		# attempt to upload to S3
		try: 
			s3.Bucket(mys3bucket).put_object(Key=dest_dir+filename, Body=data)
			return True
		except botocore.exceptions.ClientError as e:
			if e.response['Error']['Code']=='404':
				print("The bucket or directory does not exist.")
			return False
	except FileNotFoundError:
		print("[Error] Source \"%s\" does not exist."%(src_dir+filename))
		return False

	