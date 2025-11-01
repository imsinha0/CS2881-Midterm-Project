from utils.helpers import read_txt
from modules.Model import Model

model_id = "meta-llama/Llama-3.2-1B-Instruct"
prompts = read_txt("prompts/prompts.txt").split("\n")

model = Model(model_id, prompts[0])
scores = model.runner()
print(scores)

