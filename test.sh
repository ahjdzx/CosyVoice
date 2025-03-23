curl https://xxx/v1/audio/speech -H 'Content-Type: application/json' -d '{
           "model": "tts-1",
           "voice": "asset/zero_shot_prompt.wav",
           "input": "你好啊，亲爱的朋友们",
           "speed": 1.0
           }' >xx.wav
