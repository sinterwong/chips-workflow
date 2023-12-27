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
        "url": "rtsp://admin:admin@your_ip:554/MainStream"
    })
    
    time.sleep(50)
    
    # 关闭视频流
    stop_video("testVideo")
