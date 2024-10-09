import requests
import json

def fetch_wikipedia_data_via_api(title):
    # 日本語版WikipediaのAPIエンドポイント
    url = "https://ja.wikipedia.org/w/api.php"
    
    # タイトルはエンコードせずそのまま送信
    params = {
        "action": "query",
        "format": "json",
        "titles": title,  # エンコードせずにそのまま使用
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,  # このプロパティを使うためには "extracts" を指定する必要がある
    }

    # APIリクエストを送信
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # HTTPエラーが発生した場合に例外を発生させる
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTPエラーが発生しました: {http_err}")
        return None
    except Exception as err:
        print(f"エラーが発生しました: {err}")
        return None

    # レスポンスの内容を確認
    try:
        data = response.json()
        print(f"APIから取得したデータ: {data}")  # デバッグ用に表示

        pages = data['query']['pages']
        page_id = next(iter(pages))
        page = pages[page_id]

        # ページが見つかったかどうかを確認
        if "missing" in page:
            print(f"ページが見つかりません: {title}")
            return None

        # JSON形式でデータをまとめる
        result = {
            "title": page.get('title', title),  # 元の日本語タイトルを使う
            "summary": page.get('extract', '概要がありません')
        }
        return result
    except KeyError as key_err:
        print(f"必要なデータが見つかりません: {key_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"JSONデコードエラーが発生しました: {json_err}")
        return None
    except Exception as err:
        print(f"不明なエラーが発生しました: {err}")
        return None

def save_to_json(data, filename):
    if data is not None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{filename} にJSONファイルが正常に保存されました！")
    else:
        print("データがありませんでした。JSONファイルを作成できません。")

# Wikipediaページを指定してデータを取得（日本語タイトル）
title = "沼津工業高等専門学校"
wikipedia_data = fetch_wikipedia_data_via_api(title)

# データが取得できた場合、JSONファイルとして保存
save_to_json(wikipedia_data, "wikipedia_data_ja.json")
