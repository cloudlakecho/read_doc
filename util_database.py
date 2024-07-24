
"""

  util_database.py
     Stored the image to database in Cloud

  Cloud Cho from July 17, 2024

  To do
     log in the AWS account to access DynamoDB
     grep file in local computer
     find file and read from AWS

  Reference:
    Write: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/programming-with-python.html

"""

import botocore
import boto3

dynamodb = boto3.client('dynamodb')

table_name = "Face"
file_name = "/home/cloud/Pictures/Untitled.jpg"
read_option = "rb"
#
# To Do
#   how to grep file, which we don't need to close it
f_in = open(file_name, read_option)

# Test to check database is available
try:
    response = dynamodb.put_item(
      TableName = table_name,
      Item = {file_name : f_in}
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
