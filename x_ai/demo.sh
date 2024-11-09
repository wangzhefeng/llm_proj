curl https://api.x.ai/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer xai-6Dy2U57Z0v7wFVhk1ewujvwyngEVYC3jE9VdrMRf1oTu4J0C81PqSwz3GCdqjqNZpR9PZ36PW9Qwfr3O" -d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a test assistant."
    },
    {
      "role": "user",
      "content": "Testing. Just say hi and hello world and nothing else."
    }
  ],
  "model": "grok-beta",
  "stream": false,
  "temperature": 0
}'