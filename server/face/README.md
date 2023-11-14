# 人脸服务API接口文档

## 概览

此文档描述了人脸服务API的使用方法。通过这些API，您可以在人脸库中创建、更新、删除和搜索用户，以及基于目前的人脸库对视频进行实时监测，返回人脸信息。每个API调用都会返回一个JSON格式的响应。

## 基础URL
http://your_ip:9797


## 接口说明

### 1. 创建用户

创建一个新用户并将其人脸数据添加到库中。

#### 请求

POST /face/v0/facelib/createOne

#### POST 参数

- `userId` (必须): 用户的唯一标识符。
- `url` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。

#### 响应

```json
{
  "status": "OK", 
  "code": 200, 
  "message": "User was successfully created."
}
```

### 2. 更新用户
更新现有用户的人脸数据。

#### 请求
POST /face/v0/facelib/updateOne

#### POST 参数
- `userId` (必须): 用户的唯一标识符。
- `url` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。

#### 响应
```json
{
  "status": "OK", 
  "code": 200, 
  "message": "User was successfully updated."
}
```

### 3. 删除用户
从人脸库中删除一个用户。

#### 请求
DELETE /face/v0/facelib/{user_id}

#### DELETE 参数
- `userId` (必须): 用户的唯一标识符。

#### 响应
```json
{
  "status": "OK", 
  "code": 200, 
  "message": "User was successfully deleted."
}
```

### 4. 搜索用户
在人脸库中搜索匹配的用户。

#### 请求
- GET /face/v0/facelib/search?url={url}
- POST /face/v0/facelib/search

#### GET/POST 参数
- `url` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "2"  # the user"s ID number.
}
```

### 5. 两图比较
比较两个图片人脸的相似度。

#### 请求
- GET /face/v0/facelib/compare?url1={url1}&url2={url2}
- POST /face/v0/facelib/compare

#### GET 参数
- `url1` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。
- `url2` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。

#### POST 参数示例（必须为两张图片）
```json
[
    {"url": "http://xxx.com/xxx.jpg"},
    {"url": "http://xxx.com/xxx.jpg"}
]
```

### 6. 启动视频流
启动视频流监测，基于当前的人脸库进行人脸识别

#### 请求
POST /stream/startVideo

#### POST 参数
- `name` (必须): 网络视频流名称。
- `url` (必须): 网络视频流链接。

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "Video was successfully starting"
}
```

### 7. 停用视频流
根据name停用视频流

#### 请求
GET /stream/stopVideo

#### GET 参数
- `name` (必须): 网络视频流名称。

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "Video was successfully stopped"
}
```

### 错误处理
#### 重复创建id
```json
{
    "status": "ERROR", 
    "code": 500, 
    "message": "UNIQUE constraint failed: AppUser.idNumber"
}
```

#### 修改无效id
```json
{
    "status": "ERROR", 
    "code": 404, 
    "message": "User not found"
}
```

#### 删除无效id
```json
{
    "status": "ERROR", 
    "code": 404, 
    "message": "User not found"
}
```

#### 库中没有找到要查询的人脸
```json
{
    "status": "No Content", 
    "code": 204, 
    "message": "No face found"
}
```

#### 视频流id重复
```json
{
    "status": "Service Unavailable", 
    "code": 503, 
    "message": "Video startup failed."
}
```

#### 停用没有启动的视频流
```json
{
    "status": "Service Unavailable", 
    "code": 503, 
    "message": "Video stop failed"
}
```

请确保在实际应用中处理这些错误。

## 代码示例
以下是使用Python中的requests库进行API调用的代码示例。

```python
import requests
import json
import time
import base64

BASE_URL = "http://localhost:9797"  # Change this to your server's URL and port
headers = {'Content-Type': 'application/json'}

def get_base64_of_image(url):
    with open(url, "rb") as image_file:
        image_bytes = image_file.read()
    encoded_image = base64.b64encode(image_bytes)
    encoded_image_string = "data:image/jpeg;base64," + \
        encoded_image.decode("utf-8")
    return encoded_image_string


def create_user(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/createOne", data=json.dumps(data), headers=headers)
    print("Create User:", response.json())


def update_user(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/updateOne", data=json.dumps(data), headers=headers)
    print("Update User:", response.json())


def delete_user(user_id):
    response = requests.delete(f"{BASE_URL}/face/v0/facelib/{user_id}")
    print("Delete User:", response.json())


def search_user(url):
    response = requests.get(f"{BASE_URL}/face/v0/facelib/search?url={url}")
    print("Search User:", response.json())


def search_user_post(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/search", data=json.dumps(data), headers=headers)
    print("Search User:", response.json())


def compare_two_users(url1, url2):
    response = requests.get(
        f"{BASE_URL}/face/v0/facelib/compare?url1={url1}&url2={url2}")
    print("Compare User:", response.json())


def compare_two_users_post(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/compare", data=json.dumps(data), headers=headers)
    print("Compare User:", response.json())


def start_video(post_data):
    response = requests.post(
        f"{BASE_URL}/stream/startVideo", data=json.dumps(post_data), headers=headers)
    print("Start video:", response.json())


def stop_video(name):
    response = requests.get(f"{BASE_URL}/stream/stopVideo?name={name}")
    print("Stop video:", response.json())


if __name__ == "__main__":
    create_user({"userId": "12345", "url": get_base64_of_image("/path/image.png")})
    update_user({"userId": "12345", "url": get_base64_of_image("/path/image.png")})

    search_user("/path/image.png")
    search_user_post({"url": get_base64_of_image("/path/image.png")})

    compare_two_users("/path/image.png",
                      "/path/image.png")

    compare_two_users_post(([
        {"url": get_base64_of_image("/path/image.png")},
        {"url": get_base64_of_image("/path/image.png")}
    ]))

    delete_user("12345")

    start_video(
        {"name": "video1", "url": "rtsp://admin:zkfd123.com@192.168.31.31:554/Streaming/Channels/101"})
    time.sleep(10)  # 10s
    stop_video("video1")
```

## TODO
- [x] 单张人脸库增删改接口
- [x] 部分接口Post请求支持
- [x] 人脸接口支持base64编码
- [ ] 批量增删改接口
