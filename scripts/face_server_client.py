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
    create_user({"userId": "12345", "url": get_base64_of_image("image1.png")})
    update_user({"userId": "12345", "url": get_base64_of_image("image2.png")})

    search_user("/root/workspace/softwares/flowengine/image1.png")
    search_user_post({"url": get_base64_of_image("image2.png")})

    compare_two_users("/root/workspace/softwares/flowengine/image1.png",
                      "/root/workspace/softwares/flowengine/image1.png")

    compare_two_users_post(([
        {"url": get_base64_of_image("image1.png")},
        {"url": get_base64_of_image("image2.png")}
    ]))

    delete_user("12345")

    start_video(
        {"name": "video1", "url": "rtsp://admin:zkfd123.com@192.168.31.31:554/Streaming/Channels/101"})
    time.sleep(10)  # 10s
    stop_video("video1")
