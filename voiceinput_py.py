#pip install SpeechRecognition pyaudio
#pip install requests
#pip install pyaudio

import Speech_recognition as sr


# 音声をマイクから取得してテキストに変換する関数
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        # マイク入力の設定
        with mic as source:
            print("環境音を調整しています。しばらくお待ちください...")
            recognizer.adjust_for_ambient_noise(source)
            print("話してください...")
            audio = recognizer.listen(source)

        # Google Web Speech APIを使用して音声をテキストに変換
        print("音声認識中...")
        text = recognizer.recognize_google(audio, language="ja-JP")
        print(f"認識されたテキスト: {text}")
        return text

    except sr.UnknownValueError:
        print("音声が認識できませんでした。")
        return ""
    except sr.RequestError:
        print("Google Web Speech APIに接続できませんでした。")
        return ""

if __name__ == "__main__":
    text = recognize_speech_from_mic()
    if text:
        print(f"結果: {text}")
    else:
        print("音声認識に失敗しました。")
