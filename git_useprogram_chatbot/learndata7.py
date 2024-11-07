import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
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

# フィードバックデータを1行ごとに読み込む関数
def load_feedback_data(file_path="data/feedback.json"):
    feedback = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    feedback.append(json.loads(line.strip()))  # 1行ごとにJSONデコード
                except json.JSONDecodeError as e:
                    print(f"JSON デコードエラー (行スキップ): {e}")
    except FileNotFoundError:
        print("フィードバックファイルが見つかりませんでした。")
    return feedback

# フィードバックデータをベクトル化
def preprocess_feedback_embeddings(feedback_data, tokenizer, model):
    embeddings = {}
    for entry in feedback_data:
        question = entry.get("question", "")
        bot_answer = entry.get("bot_answer", "")
        # 'human_answer' が存在しない場合は 'bot_answer' を使用
        human_answer = entry.get("human_answer", bot_answer)
        inputs = tokenizer(question, return_tensors="pt")
        embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        embeddings[question] = (embedding, human_answer)
    return embeddings

# 質問とフィードバック・Wikipediaタグを比較して最も類似した質問を探す
def find_similar_question(question, feedback_embeddings, tokenizer, model, data, recommendation_data, threshold=0.7):
    # 質問の埋め込みを計算
    inputs = tokenizer(question, return_tensors="pt")
    question_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    max_similarity = threshold
    best_answer = None

    # フィードバックデータ内の類似度計算
    print("\n--- フィードバックデータとの類似度計算 ---")
    for feedback_question, (feedback_embedding, feedback_answer) in feedback_embeddings.items():
        similarity = cosine_similarity(question_embedding, feedback_embedding)[0][0]
        print(f"フィードバック質問: '{feedback_question}' の類似度: {similarity:.4f}")
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = feedback_answer

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

# フィードバックを保存する関数
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
    feedback_data = load_feedback_data("data/feedback.json")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
    embedding_model = AutoModel.from_pretrained("xlm-roberta-base")

    feedback_embeddings = preprocess_feedback_embeddings(feedback_data, tokenizer, embedding_model)

    print("沼津高専チャットボットへようこそ！'exit'と入力して終了できます。")

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            break

        bot_answer = find_similar_question(user_input, feedback_embeddings, tokenizer, embedding_model, data, recommendation_data)
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
