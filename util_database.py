
"""

  util_database.py
     Stored the image to database in Cloud

  Cloud Cho from July 17, 2024


  To do
     log in the AWS account to access DynamoDB
     grep file in local computer
     find file and read from AWS

  Work? - no

  Runtime enivronment
     set ~/.aws/config
     set ~/.aws/credentials
     use "Config" object at "client" function

  Reference:
    Write: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html
    Runtime setting: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
"""

import botocore
import boto3
from botocore.config import Config


#
# Error
#   region missing
#
region = "us-west-1"
table_name = "Face"
file_name = "/home/cloud/Pictures/Untitled.jpg"
read_option = "rb"

my_config = Config(
    region_name = region,
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)

dynamodb = boto3.client('dynamodb', config=my_config)

#
# To Do
#   how to grep file, which we don't need to close it
img_file = request.session.get(file_name)
str_ = json.dumps(str(img_file))


# Check existing tables
db = session.resource('dynamodb', region_name="us-east-2")
tables = list(db.tables.all())
print(tables)


# Test to check database is available
try:
    response = dynamodb.put_item(
      TableName = table_name,
      Item = {file_name : str_}
    )
except botocore.exceptions.ClientError as err:
    print('Error Code: {}'.format(err.response['Error']['Code']))
    print('Error Message: {}'.format(err.response['Error']['Message']))
    print('Http Code: {}'.format(err.response['ResponseMetadata']['HTTPStatusCode']))
    print('Request ID: {}'.format(err.response['ResponseMetadata']['RequestId']))

    if err.response['Error']['Code'] in ('ProvisionedThroughputExceededException', 'ThrottlingException'):
        print("Received a throttle")
    elif err.response['Error']['Code'] == 'InternalServerError':
        print("Received a server error")
    else:
        raise err

try:
    dynamodb.put_item(
        TableName='YourTableName',
        Item={
            'pk': {'S': 'id#1'},
            'sk': {'S': 'cart#123'},
            'name': {'S': 'SomeName'},
            'inventory': {'N': '500'},
        # ... more attributes ...
        }
    )
except botocore.exceptions.ClientError as err:
    print('Error Code: {}'.format(err.response['Error']['Code']))
    print('Error Message: {}'.format(err.response['Error']['Message']))
    print('Http Code: {}'.format(err.response['ResponseMetadata']['HTTPStatusCode']))
    print('Request ID: {}'.format(err.response['ResponseMetadata']['RequestId']))

    if err.response['Error']['Code'] in ('ProvisionedThroughputExceededException', 'ThrottlingException'):
        print("Received a throttle")
    elif err.response['Error']['Code'] == 'InternalServerError':
        print("Received a server error")
    else:
        raise err


# Read
#   ref: https://stackoverflow.com/a/63012418/5595995
import boto3
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource('dynamodb', region_name=region)
table = dynamodb.Table('<TableName>')

response = table.query(
    IndexName='<Index>',
    KeyConditionExpression=Key('<key1>').eq('<value>') & Key('<key2>').eq('<value>'),
    FilterExpression=Attr('<attr>').eq('<value>')
)

print(response['Items'])
