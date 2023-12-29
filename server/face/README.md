# 人脸库服务API接口文档

## 概览

此文档描述了人脸库服务API的使用方法。通过这些API，您可以在人脸库中创建、更新、删除和搜索用户。每个API调用都会返回一个JSON格式的响应。

## 基础URL
http://your_ip:9797


## 接口说明

### 1. 创建用户

创建一个新用户并将其人脸数据添加到库中。

#### 请求

POST /face/v0/facelib/createOne

#### POST 参数

- `userId` (必须): 用户的唯一标识符。
- `libName` (必须): 该操作所在人脸库名称。
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
- `libName` (必须): 该操作所在人脸库名称。
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
- GET /face/v0/facelib/search?url={url}&libName={libName}
- POST /face/v0/facelib/search

#### GET/POST 参数
- `url` (必须): 包含用户人脸图片的路径（支持http、本地图片路径和base64编码）。
- `libName` (必须): 待搜索的人脸库名称。

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "2" # the user's ID number.
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

### 6. 批量创建用户
批量创建用户并将其人脸数据添加到库中。

#### 请求
POST /face/v0/facelib/createBatch

#### POST 参数示例
```json
[
    {"userId": "11111", "libName": "temp", "url": "http://xxx.com/xxx.jpg"},
    {"userId": "22222", "libName": "temp", "url": "http://xxx.com/xxx.jpg"}
]
```

#### 响应
```json
{
  "status": "OK", 
  "code": 200, 
  "message": "Users were successfully created."
}
```

### 7. 批量更新用户
批量更新现有用户的人脸数据。

#### 请求
POST /face/v0/facelib/updateBatch

#### POST 参数示例
```json
[
    {"userId": "11111", "libName": "temp", "url": "http://xxx.com/xxx.jpg"},
    {"userId": "22222", "libName": "temp", "url": "http://xxx.com/xxx.jpg"}
]
```

#### 响应
```json
{
  "status": "OK", 
  "code": 200, 
  "message": "Users were successfully updated."
}
```

### 8. 批量删除用户
从人脸库中批量删除用户。

#### 请求
POST /face/v0/facelib/deleteBatch

#### POST 参数示例
```json
[
    {"userId": "11111"},
    {"userId": "22222"}
]
```

#### 响应
```json
{
  "status": "OK", 
  "code": 200, 
  "message": "Users were successfully deleted."
}
```

### 9. 人脸质检
对人脸图片进行质检。

#### 请求
POST /face/v0/facelib/faceQuality

#### POST 参数示例
```json
{"name": "temp", "url": "http://xxx.com/xxx.jpg"}
```

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "0"  # 质检结果，0：正常，1：图片尺寸过小，2：长宽比过大，3：模糊度或亮度过大，4：人脸角度过大，5：大胡子，6：普通眼镜，7：口罩遮挡，8：墨镜，9：其它遮挡，-1：无人脸或未知错误
}
```

### 10. 人脸识别视频监控
对视频流进行人脸识别。

#### 请求
POST /face/v0/stream/startVideo

#### POST 参数示例
```json
{
    "name": "testVideo", 
    "libName": "testdb", 
    "url": "rtsp://admin:admin@your_ip:554/MainStream",
    "interfaceUrl": "http://your_ip/:8082/v1/flow/faceRecognizeResult"
}
```

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "Video was successfully starting"
}
```

### 11. 关闭视频流
关闭视频流。

#### 请求
GET /face/v0/stream/stopVideo?name={name}

#### GET 参数
- `name` (必须): 视频流名称。

#### 响应
```json
{
    "status": "OK", 
    "code": 200, 
    "message": "Video was successfully stopping"
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

#### 使用无效id
```json
{
    "status": "ERROR", 
    "code": 404, 
    "message": "User not found"
}
```

#### 批量操作部分失败
```json
{
    "status": "Partial Content", 
    "code": 206, 
    "message": "
        Some users failed.\n
        Algorithm failed: 1111, 2222\n
        Database failed: : 3333, 4444\n
    "
}
```

#### 批量操作全部失败
```json
{
    "status": "Service Unavailable", 
    "code": 503, 
    "message": "All users failed to create."
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

请确保在实际应用中处理这些错误。

## 代码示例
以下是使用Python中的requests库进行API调用的代码示例。

```python
import requests
import json
import base64
import time

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


def create_batch_users(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/createBatch", data=json.dumps(data), headers=headers)
    print("Create Batch Users:", response.json())


def update_user(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/updateOne", data=json.dumps(data), headers=headers)
    print("Update User:", response.json())


def update_batch_users(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/updateBatch", data=json.dumps(data), headers=headers)
    print("Update Batch Users:", response.json())


def delete_user(user_id):
    response = requests.delete(f"{BASE_URL}/face/v0/facelib/{user_id}")
    print("Delete User:", response.json())


def delete_batch_users(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/deleteBatch", data=json.dumps(data), headers=headers)
    print("Delete Batch Users:", response.json())


def search_user(lname, url):
    response = requests.get(
        f"{BASE_URL}/face/v0/facelib/search?url={url}&libName={lname}")
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


def face_quality(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/facelib/faceQuality", data=json.dumps(data), headers=headers)
    print("Face Quality:", response.json())


def start_video(data):
    response = requests.post(
        f"{BASE_URL}/face/v0/stream/startVideo", data=json.dumps(data), headers=headers)
    print("Start video:", response.json())


def stop_video(name):
    response = requests.get(
        f"{BASE_URL}/face/v0/stream/stopVideo?name={name}", headers=headers)
    print("Stop video:", response.json())


if __name__ == "__main__":
    create_user({"userId": "12345", "libName": "temp1",
                "url": get_base64_of_image("image1.png")})
    create_batch_users(([
        {"userId": "11111", "libName": "temp2",
            "url": get_base64_of_image("image1.png")},
        {"userId": "22222", "libName": "temp2",
            "url": get_base64_of_image("image2.png")}
    ]))

    # 单个更新允许换库
    update_user({"userId": "12345", "libName": "temp1",
                "url": get_base64_of_image("image2.png")})

    # 批量操作暂时不允许“换库”
    update_batch_users(([
        {"userId": "11111", "libName": "temp2",
            "url": get_base64_of_image("image2.png")},
        {"userId": "22222", "libName": "temp2",
            "url": get_base64_of_image("image1.png")}
    ]))
    
    search_user("temp1", "/path/image2.png")
    search_user_post(
        {"name": "temp2", "url": get_base64_of_image("image2.png")})

    compare_two_users("/path/image1.png",
                      "/path/image1.png")

    compare_two_users_post(([
        {"url": get_base64_of_image("image1.png")},
        {"url": get_base64_of_image("image2.png")}
    ]))

    delete_user("12345")
    delete_batch_users(([
        {"userId": "11111"},
        {"userId": "22222"}
    ]))

    # 质检结果，0：正常，1：图片尺寸过小，2：长宽比过大，3：模糊度或亮度过大，4：人脸角度过大，5：大胡子，6：普通眼镜，7：口罩遮挡，8：墨镜，9：其它遮挡，-1：无人脸或未知错误
    face_quality({"name": "temp2", "url": get_base64_of_image("image2.png")})
    
    # 开启视频流
    start_video({
        "name": "testVideo", 
        "libName": "testdb", 
        "url": "rtsp://admin:admin@your_ip:554/MainStream", 
        "interfaceUrl": "http://your_ip/:8082/v1/flow/faceRecognizeResult"
    })
    
    time.sleep(50)
    
    # 关闭视频流
    stop_video("testVideo")

```

## TODO
- [x] 单张人脸库增删改接口
- [x] 部分接口Post请求支持
- [x] 人脸接口支持base64编码
- [x] 批量增删改接口
- [x] 人脸库分库增删改
- [x] 人脸质检接口
- [x] 人脸视频流识别接口（开启、关闭）
- [ ] 活体检测功能
- [ ] 批量搜索
- [ ] 批量质检
