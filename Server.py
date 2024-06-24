import socket
from transformers import MarianMTModel, MarianTokenizer
import torch

def translate_text(input_text, source_lang, target_lang, max_length=50):
    # Load pre-trained MarianMT model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        translated_ids = model.generate(input_ids, max_length=max_length)
    
    # Decode the translated text
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True, max_length=max_length)
    return translated_text

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 9999))

server.listen()

client, addr = server.accept()

done = False

while not done:
    msg = client.recv(1024).decode('utf-8')
    if msg == 'quit':
        done = True
    else:
        print(msg)
    input_text = input("Text: ")
    source_lang = "fr" 
    target_lang = "en"
    translated_text = translate_text(input_text, source_lang, target_lang, max_length=100)
    client.send((translated_text).encode('utf-8'))

client.close()
server.close()
