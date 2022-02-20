import os
import openai

openai.api_key = "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"

response = openai.Completion.create(
  engine="text-davinci-001",
  prompt="Create an outline for an essay about Walt Disney and his contributions to animation:\n\nI:",
  temperature=0,
  max_tokens=64,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(response.choices[0].text)