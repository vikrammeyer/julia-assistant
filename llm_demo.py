# from transformers import T5Tokenizer, T5ForConditionalGeneration
from time import time

# start = time()
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
# print('models loaded in ', time() - start)

# start = time()
# input_text = "generate julia code for searching through a vector of strings and returning the strings that contain the word mango"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
# print('generated response in ', time() - start)

from transformers import AutoTokenizer, AutoModelForCausalLM

start = time()
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
print('models loaded in ', time() - start)

start = time()
input_text = "generate julia code for searching through a vector of strings and returning the strings that contain the word mango"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
print('generated response in ', time() - start)