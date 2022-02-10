from profanity_check import predict, predict_prob

text = "This is a test"
# result = predict(['fuck you'])
result = predict([text])
if result[0] == 0:
    output = "No profanity"
else:
    output = "Profanity"
    
print(output)