# test uploading file
#
#  reference:
#    https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
#    (File Listing) https://qiita.com/koji4104/items/15ac578e561f53a0dadc

import sys
import os
import boto3

BUCKET_NAME = 'face-recognition-us-east'

s3 = boto3.resource('s3')

bucket = s3.Bucket(BUCKET_NAME)

# this is comment


# file listing
objs = bucket.meta.client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='')

keycount = objs.get('KeyCount')

print('number of objects: %d'%(keycount))

if keycount > 0:
    # if keycount > 0, then show all files
    for o in objs.get('Contents'):
        key  = o.get('Key')
        size = o.get('Size')
        date = o.get('Date')

        print('%s\t%s\t%s'%(key,size,date))

if len(sys.argv) < 2:
    print('usage: %s [upload_file_name]'%(sys.argv[0]))
    sys.exit(0)

# translate folder name from \ to /
keyname = sys.argv[1].translate(str.maketrans('\\','/'))
	
if keycount > 0:
    # if keycount > 0, then show all files
    for o in objs.get('Contents'):
        key  = o.get('Key')
        size = o.get('Size')
        date = o.get('Date')

        if key == keyname:
            print('file is already uploaded to s3. skip uploading.')
            sys.exit(1)    
    
# open datafile
print('uploading file..')
if os.path.isfile(sys.argv[1]) == False:
    print('cannot find ',sys.argv[1])
    sys.exit(0)

data = open(sys.argv[1],'rb')

print('amazon s3 filename: ' + keyname)

bucket.put_object(Key=keyname, Body=data)
print('finished.')

