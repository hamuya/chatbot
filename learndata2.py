import json
from transformers import pipeline

# JSONファイルを読み込む
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 質問応答モデルを使って、最適な回答を選ぶ
def find_answer(question, data, qa_model):
    # データ全体を1つのコンテンツとして扱う
    context = ""
    for section, content in data.items():
        context += f"{section}: {content}\n"
    
    # 質問とコンテキストをモデルに渡して回答を生成
    result = qa_model(question=question, context=context)
    return result['answer']

# チャットボットの会話ループ
def chatbot():
    data = load_json_data('wikipedia_sections.json')
    
    # 日本語対応のRoBERTa質問応答モデルを使用
    qa_model = pipeline('question-answering', model='rinna/japanese-roberta-base')

    print("沼津高専チャットボットへようこそ！'exit'と入力して終了できます。")

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            break
        response = find_answer(user_input, data, qa_model)
        print(f"チャットボット: {response}")

# チャットボットを起動
if __name__ == "__main__":
    chatbot()
