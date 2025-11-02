from utils.helpers import read_txt
from modules.Model import Model
import json


model1_id = "meta-llama/Llama-3.2-1B-Instruct"
model2_id = "meta-llama/Llama-3.2-3B-Instruct"
model3_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model4_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
models = [model1_id, model2_id, model3_id, model4_id]
prompts = read_txt("prompts/prompts.txt").split("\n")

results1 = {}
for model_id in models:
    model = Model(model_id, prompts)
    scores = model.runner()
    results1[model_id] = scores

results2 = {}

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

for num_chunks in [1, 2, 4, 8, 16]:
    model = Model(model_id, prompts, num_chunks=num_chunks)
    scores = model.runner()
    results2[num_chunks] = scores


results3 = {}

for adverserial_prompt_num in [1, 2, 3]:
    model = Model(model_id, prompts, adverserial_prompt_num=adverserial_prompt_num)
    scores = model.runner()
    results3[adverserial_prompt_num] = scores

print(results1)
print(results2)
print(results3)

#print all results
#save all results to json files
with open("results1.json", "w") as f:
    json.dump(results1, f)
with open("results2.json", "w") as f:
    json.dump(results2, f)
with open("results3.json", "w") as f:
    json.dump(results3, f)
