import requests
from bs4 import BeautifulSoup
import json

def fetch_wikipedia_sections(title):
    # WikipediaページのURL
    url = f"https://ja.wikipedia.org/wiki/{title}"

    # WikipediaページのHTMLを取得
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"ページの取得に失敗しました: {response.status_code}")
        return None

    # BeautifulSoupでHTMLを解析
    soup = BeautifulSoup(response.content, 'html.parser')

    # ページ内のセクションタイトルと内容を抽出
    sections = {}
    current_section = None

    # Wikipediaのセクションはh2やh3タグで表示されているため、それを抽出
    for tag in soup.find_all(['h2', 'h3', 'p']):
        # セクションのタイトル（h2, h3）を取得
        if tag.name in ['h2', 'h3']:
            current_section = tag.text.strip().replace("[編集]", "")  # [編集]を削除
            sections[current_section] = ""
        elif current_section and tag.name == 'p':  # 段落（pタグ）を現在のセクションに追加
            sections[current_section] += tag.text.strip()

    return sections

# JSONファイルとして保存する関数
def save_to_json(data, filename):
    if data:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{filename} にJSONファイルが正常に保存されました！")
    else:
        print("データがありませんでした。JSONファイルを作成できません。")

# Wikipediaページからセクションデータを取得
title = "沼津工業高等専門学校"
sections_data = fetch_wikipedia_sections(title)

# データが取得できた場合、JSONファイルとして保存
save_to_json(sections_data, "wikipedia_sections.json")
