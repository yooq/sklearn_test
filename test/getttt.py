import parser
from urllib.request import urlopen

import requests
import json
import urllib.parse as parse

url ='http://127.0.0.1:5000/test_1.0'

data ={'name':'xiaoming','age':'18'}

pos =requests.post(url,json=data)  #text  {"return_code": "200", "return_info": "处理成功", "result": "xiaoming今年18岁"}


a = parse.urlencode({"name":"xiaoming"})
r= requests.Request(url,a)
print(r)  #<Request [http://127.0.0.1:5000/test_1.0]>
response = urlopen(r)  #
# print(response)

