import time
import base64
import requests


BASE_URL = "http:0.0.0.0:7006"


def db_info():
    """
    curl --header "Content-Type: application/json" --request POST --data '{"db_name":""}' http://0.0.0.0:7006/v1/db/info
    """
    since = time.time()
    url = "{}/v1/db/info".format(BASE_URL)
    data = {
        "db_name": ""
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since
    print('db_info use time : {}s'.format(use))


def db_create():
    since = time.time()
    url = "{}/v1/db/create".format(BASE_URL)
    data = {
        "db_name": "temp_test",
        "max_size": 100,
        "info": "test"
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since 
    print('db_create use time : {}s'.format(use))


def db_remove():
    since = time.time()
    url = "{}/v1/db/remove".format(BASE_URL)
    data = {
        "db_name": "temp_test"
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since
    print("db_remove use {}s".format(use))


def db_insert():
    since = time.time()
    url = "{}/v1/db/insert".format(BASE_URL)
    with open('./data/sample_1.jpg', 'rb') as f:
        image = base64.b64encode(f.read())
    data = {
        "db_name": "temp_test",
        "feature_id": 1,
        "image": image.decode('ascii')
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since
    print("db_insert use {}s".format(use))


def db_delete():
    since = time.time()
    url = "{}/v1/db/remove".format(BASE_URL)
    data = {
        "db_name": "temp_test",
        "feature_id": 1
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since
    print("db_remove use {}s".format(use))


def db_search():
    since = time.time()
    url = "{}/v1/db/search".format(BASE_URL)
    with open('./data/sample_1.jpg', 'rb') as f:
        image = base64.b64encode(f.read())
    data = {
        "db_name": "temp_test",
        "image": image.decode('ascii')
    }
    response = requests.post(url=url, json=data).json()
    print(response)
    use = time.time() - since
    print("db_search use {}s".format(use))


if __name__ == "__main__":
    db_info()
    # db_create()
    # db_insert()
    # db_remove()
    # db_search()
    # db_delete()


