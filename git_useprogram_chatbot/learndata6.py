import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# JSONファイルを読み込む
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"{file_path} が見つかりませんでした。")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON デコードエラー ({file_path}):", e)
        return {}

# 質問とWikipediaタグを比較して最も類似した質問を探す
def find_similar_question(question, tokenizer, model, data, recommendation_data, threshold=0.7):
    # 質問の埋め込みを計算
    inputs = tokenizer(question, return_tensors="pt")
    question_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    max_similarity = threshold
    best_answer = None

    # Wikipediaデータ内のタグとコンテンツとの類似度計算
    print("\n--- Wikipediaデータとの類似度計算 ---")
    for section, content in data.items():
        section_inputs = tokenizer(section, return_tensors="pt", truncation=True)
        section_embedding = model(**section_inputs).last_hidden_state.mean(dim=1).detach().numpy()
        section_similarity = cosine_similarity(question_embedding, section_embedding)[0][0]
        print(f"タグ '{section}' の類似度: {section_similarity:.4f}")

        content_inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        content_embedding = model(**content_inputs).last_hidden_state.mean(dim=1).detach().numpy()
        content_similarity = cosine_similarity(question_embedding, content_embedding)[0][0]
        print(f"セクション: '{section}' の内容との類似度: {content_similarity:.4f}")

        if section_similarity > max_similarity or content_similarity > max_similarity:
            max_similarity = max(section_similarity, content_similarity)
            best_answer = content

    # 推薦選抜データ内のタグとコンテンツとの類似度計算
    print("\n--- 推薦選抜データとの類似度計算 ---")
    for section, content in recommendation_data.items():
        section_inputs = tokenizer(section, return_tensors="pt", truncation=True)
        section_embedding = model(**section_inputs).last_hidden_state.mean(dim=1).detach().numpy()
        section_similarity = cosine_similarity(question_embedding, section_embedding)[0][0]
        print(f"タグ '{section}' の類似度: {section_similarity:.4f}")

        content_inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        content_embedding = model(**content_inputs).last_hidden_state.mean(dim=1).detach().numpy()
        content_similarity = cosine_similarity(question_embedding, content_embedding)[0][0]
        print(f"セクション: '{section}' の内容との類似度: {content_similarity:.4f}")

        if section_similarity > max_similarity or content_similarity > max_similarity:
            max_similarity = max(section_similarity, content_similarity)
            best_answer = content

    print(f"\n最も高い類似度: {max_similarity:.4f}")
    return best_answer

# フィードバックを保存する関数（最終的な回答生成には使用しない）
def save_feedback(question, bot_answer, rating, human_answer=None, file_path="data/feedback.json"):
    feedback_data = {
        "question": question,
        "bot_answer": bot_answer,
        "rating": rating,
        "human_answer": human_answer
    }

    with open(file_path, "a", encoding="utf-8") as file:
        json.dump(feedback_data, file, ensure_ascii=False)
        file.write("\n")

# チャットボット
def chatbot():
    data = load_json_data('data/wikipedia_sections.json')
    recommendation_data = load_json_data('data/numazu_recommendation_selection_retry.json')

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
    qa_model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
    embedding_model = AutoModel.from_pretrained("xlm-roberta-base")

    print("沼津高専チャットボットへようこそ！'exit'と入力して終了できます。")

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            break

        bot_answer = find_similar_question(user_input, tokenizer, embedding_model, data, recommendation_data)
        print(f"あなたの質問: {user_input}")
        print(f"チャットボットの回答: {bot_answer}")

        try:
            rating = int(input("評価を1から5で入力してください（1が最低、5が最高）: "))
        except ValueError:
            print("無効な入力です。1から5の数字で評価してください。")
            continue

        human_answer = None
        if rating < 3:
            human_answer = input("改善された回答を入力してください: ")
        
        save_feedback(user_input, bot_answer, rating, human_answer)

# チャットボットを起動
if __name__ == "__main__":
    chatbot()
