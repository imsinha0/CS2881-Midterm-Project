from utils.helpers import read_txt
from modules.Model import Model
import json

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"



prompts = read_txt("prompts/prompts.txt").split("\n")

model = Model(model_id, prompts, inject_noise=True)
scores = model.runner()
print(scores)

'''

prompts1 = read_txt("prompts/harvard_answerable.txt").split("\n")
prompts2 = read_txt("prompts/harvard_unanswerable.txt").split("\n")

model = Model(model_id, prompts1)
scores1 = model.runner()

model = Model(model_id, prompts2)
scores2 = model.runner()

print(scores1)
print(scores2)
'''