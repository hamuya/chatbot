import json
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# JSONファイルを読み込む
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# フィードバックデータを読み込む関数
def load_feedback_data(file_path="feedback.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            feedback = [json.loads(line) for line in file]
        return feedback
    except FileNotFoundError:
        return []

# フィードバックデータを事前にベクトル化して保存
def preprocess_feedback_embeddings(feedback_data, tokenizer, model):
    embeddings = {}
    for entry in feedback_data:
        question = entry["question"]
        inputs = tokenizer(question, return_tensors="pt")
        embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        embeddings[question] = (embedding, entry["human_answer"] or entry["bot_answer"])
    return embeddings

# 類似質問を探す関数（事前計算済みの埋め込みを利用）
def find_similar_question(question, feedback_embeddings, tokenizer, model, threshold=0.7):
    # 入力された質問をベクトル化
    inputs = tokenizer(question, return_tensors="pt")
    question_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    # フィードバック内のベクトルと類似度を比較
    for feedback_question, (feedback_embedding, feedback_answer) in feedback_embeddings.items():
        similarity = cosine_similarity(question_embedding, feedback_embedding)

        # 類似度が閾値を超えた場合、その回答を返す
        if similarity >= threshold:
            return feedback_answer
    return None

# フィードバックを保存し、埋め込みを更新する関数
def save_and_update_feedback(question, bot_answer, rating, feedback_embeddings, tokenizer, model, human_answer=None):
    feedback_data = {
        "question": question,
        "bot_answer": bot_answer,
        "rating": rating,
        "human_answer": human_answer
    }

    # フィードバックデータをファイルに追加保存
    with open("feedback.json", "a", encoding="utf-8") as file:
        json.dump(feedback_data, file, ensure_ascii=False, indent=4)
        file.write("\n")

    # 新しいフィードバックを埋め込みに追加
    inputs = tokenizer(question, return_tensors="pt")
    new_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    feedback_embeddings[question] = (new_embedding, human_answer or bot_answer)

# 質問応答モデルを使って、最適な回答を選ぶ
def find_answer(question, data, feedback_embeddings, qa_model, tokenizer, embedding_model):
    # まずフィードバックデータから類似質問を探す
    similar_answer = find_similar_question(question, feedback_embeddings, tokenizer, embedding_model)
    if similar_answer:
        return similar_answer

    # 類似質問が見つからない場合は通常のデータで回答生成
    context = "".join([f"{section}: {content}\n" for section, content in data.items()])
    result = qa_model(question=question, context=context)
    return result['answer']

# チャットボットの会話と評価ループ
def chatbot():
    data = load_json_data('wikipedia_sections.json')
    feedback_data = load_feedback_data()

    # トークナイザとモデルをxlm-roberta-baseで読み込み
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
    qa_model = AutoModelForQuestionAnswering.from_pretrained("xlm-roberta-base")
    embedding_model = AutoModel.from_pretrained("xlm-roberta-base")  # 同じモデルを埋め込み生成にも利用

    # 質問応答パイプラインを設定
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=tokenizer, framework="pt")

    # フィードバックデータの事前ベクトル化
    feedback_embeddings = preprocess_feedback_embeddings(feedback_data, tokenizer, embedding_model)

    print("沼津高専チャットボットへようこそ！'exit'と入力して終了できます。")

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == "exit":
            break

        # モデルで回答を生成
        bot_answer = find_answer(user_input, data, feedback_embeddings, qa_pipeline, tokenizer, embedding_model)
        print(f"チャットボット: {bot_answer}")

        # 回答に対する評価をユーザーに求める
        try:
            rating = int(input("評価を1から5で入力してください（1が最低、5が最高）: "))
        except ValueError:
            print("無効な入力です。1から5の数字で評価してください。")
            continue

        # 低評価の場合、手動で修正回答を入力
        human_answer = None
        if rating < 3:
            human_answer = input("改善された回答を入力してください: ")
        
        # 評価とフィードバックを保存、埋め込み更新
        save_and_update_feedback(user_input, bot_answer, rating, feedback_embeddings, tokenizer, embedding_model, human_answer)

# チャットボットを起動
if __name__ == "__main__":
    chatbot()
