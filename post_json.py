import urllib.request, json

if __name__ == '__main__':
    url = "http://172.23.19.231:8888/get_reply" 
    method = "POST"
    headers = {"Content-Type" : "application/json"}

    # PythonオブジェクトをJSONに変換する
    obj = {
    'note': 'info or discription, anything ',
    'messages': [
        {
            'role': "システム",
            'content': 'こんにちは、りんなです。私は大規模言語モデルなのでどんな質問でも答えます。お気軽にどうぞ！'
        },
        {
            'role': "ユーザー",
            'content': 'こんにちは。私はりんごが食べたいです。美味しいりんごの見分け方を教えてください。'
        },
    ]
}
    json_data = json.dumps(obj).encode("utf-8")

    # httpリクエストを準備してPOST
    request = urllib.request.Request(url, data=json_data, method=method, headers=headers)
    with urllib.request.urlopen(request) as response:
        response_body = response.read().decode("utf-8")
        result_objs = json.loads(response_body)
        print(result_objs)