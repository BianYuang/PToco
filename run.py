import os
import re
import time
import torch
import json
from torch import nn
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
# todo
model_paths = [
    "/data2/bianyuang/models/qwen/Qwen2-7B-chat",
    "/data2/bianyuang/models/LLM-Research/Meta-Llama-3-8B-chat",
    "/data2/bianyuang/models/ZhipuAI/glm-4-9b-chat",
    "/data2/bianyuang/models/LLM-Research/gemma-2-9b-base",
]
device_ids = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"] 

now = datetime.now()
current_date = str(now.date())
current_time = str(now.time())[:8]
time_info = current_date + '_' + current_time + '---'
setting = f'{time_info}aqua_20_clip' ################ todo
prob_clip = 0.2 # todo
top_k = 5
max_length = 400 # todo
test_file = '/data2/bianyuang/cooperate_experiment/AQuA/data/test_co.json' # todo

model_name_list = [i.split('/')[-1] for i in model_paths]
result_file_name = f'{setting}---' + '+'.join(model_name_list) + '.txt'
result_file_path = os.path.join('/data2/bianyuang/cooperate_experiment/AQuA/result', result_file_name) # todo

load_models = [
    AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True).to(device_ids[i])
    for i, name in enumerate(model_paths)
]
load_tokenizers = [AutoTokenizer.from_pretrained(name, trust_remote_code=True) for name in model_paths]

# todo
example = '''Below are some examples on Q&A about algebra. Follow the format in the examples and answer the question at the end. Briefly analyze first and then include your final choice in {{}}.
Question: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices:
A. 50 B. 45 C. 65 D. 78 E. 64
Answer: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. So the answer is {{A}}.

Question: If a / b = 3/4 and 8a + 5b = 22,then find the value of a.
Answer Choices:
A. 1/2 B. 3/2 C. 5/2 D. 4/2 E. 7/2
Answer: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. So the answer is {{B}}.

Question: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?
Answer Choices:
A. 53 km B. 55 km C. 52 km D. 60 km E. 50 km
Answer: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. So the answer is {{E}}.

Question: {input}
Answer: '''

models = load_models
tokenizers = load_tokenizers
def form_cluster(all_model_topk):
    all_tokens = []
    for token_dict in all_model_topk:
        all_tokens.extend(token_dict.keys())

    # Create a 2D matrix to store startswith relationships
    matrix = [[False] * len(all_tokens) for _ in range(len(all_tokens))]

    # Populate the matrix using startswith
    for i, token_a in enumerate(all_tokens):
        for j, token_b in enumerate(all_tokens):
            if token_a != token_b and token_b.startswith(token_a):
                matrix[i][j] = True

    # Identify atomic tokens by checking if a column is all False
    atomic_tokens = set(token for j, token in enumerate(all_tokens) if not any(matrix[i][j] for i in range(len(all_tokens))))

    # Create the final output dictionary
    cluster = {token: {} for token in atomic_tokens}

    # Populate the output dictionary
    for model_index, token_dict in enumerate(all_model_topk):
        for token, prob in token_dict.items():
            for atomic_token in atomic_tokens:
                if token.startswith(atomic_token):
                    if model_index not in cluster[atomic_token]:
                        cluster[atomic_token][model_index] = {}
                    cluster[atomic_token][model_index][token] = prob
    return cluster

def get_full_probabilities(model_idx, text):
    model = models[model_idx]
    tokenizer = tokenizers[model_idx]
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device_ids[model_idx])
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs[0].detach().cpu()

def get_target_model_probabilities(text, target_model_idx):
    with ThreadPoolExecutor(max_workers=len(target_model_idx)) as executor:
        futures = [executor.submit(get_full_probabilities, model_idx, text) for model_idx in target_model_idx]
        results = [future.result() for future in futures]
    return results

def collaborative_generate(prompt, max_length=20):
    model_num = len(models)
    generated_text = ''
    cur_token_dict = {model_idx: [] for model_idx in range(len(models))}
    for round_idx in range(max_length):
        start_time = time.perf_counter()
        print(f'\nround_idx:{round_idx}')
        f.write(f'\nround_idx:{round_idx}' + '\n')
        whole_text = prompt + generated_text

        target_model_idx = [i for i in range(model_num) if not cur_token_dict[i]]
        print("target_model_idx:", target_model_idx)
        f.write("target_model_idx:"+str(target_model_idx)+'\n')
        target_model_probs = get_target_model_probabilities(whole_text, target_model_idx)

        for model_idx, full_probs in zip(target_model_idx, target_model_probs):
            top_probs, top_indices = torch.topk(full_probs, top_k)
            top_tokens = [tokenizers[model_idx].decode([idx]) for idx in top_indices]
            model_topk_dict = {token: round(prob.item(), 8) for token, prob in zip(top_tokens, top_probs) if (token != ' ' and prob > prob_clip) or (token == ' ' and prob >= 0.3)}
            if not model_topk_dict:
                model_topk_dict = {token: round(prob.item(), 8) for token, prob in zip(top_tokens, top_probs)}
                max_key = max(model_topk_dict, key=model_topk_dict.get)
                model_topk_dict = {max_key: model_topk_dict[max_key]}
            cur_token_dict[model_idx] = model_topk_dict
        all_model_topk = [cur_token_dict[model_idx] for model_idx in range(model_num)]
        clusters = form_cluster(all_model_topk)
        f.write('all_model_topk:\n')
        pprint(all_model_topk)
        pprint(all_model_topk, stream=f)
        pprint(clusters)
        f.write('clusters:\n')
        pprint(clusters, stream=f)

        cur_token = None 
        max_sum = 0
        for key, model_dict in clusters.items():
            current_sum = 0
            for model, probabilities in model_dict.items():
                for token, prob in probabilities.items():
                    current_sum += prob
            if current_sum > max_sum:
                max_sum = current_sum
                cur_token = key

        max_tokens = []
        for d in all_model_topk:
            max_token = max(d, key=d.get)
            max_tokens.append(max_token)
        all_same = all(token == max_tokens[0] for token in max_tokens)
        if all_same:
            cur_token = max_tokens[0]
        
        end_time = time.perf_counter()
        run_time = end_time - start_time
        generated_text += cur_token
        f.write('token_this_round:\n' + cur_token + '\n')
        f.write(f'time:{run_time}' + '\n')
        print(generated_text)
        print(f'time:{run_time}')
        f.write('generated_text:\n' + generated_text + '\n')
        
        choice_str_idx = generated_text.find('answer is')
        choice_str = ''
        if choice_str_idx != -1:
            choice_str = generated_text[choice_str_idx:]
        if (choice_str != '' and '}' in choice_str): 
            break
        
        if not all_same:
            cur_token_dict = clusters[cur_token]
            cur_token_len = len(cur_token)
            for model_idx in range(len(models)):
                probabilities = cur_token_dict.get(model_idx, {})
                new_probabilities = {}
                prob_sum = max(1e-10, sum(probabilities.values()))
                scaling_factor = 1 / (prob_sum)

                for token, prob in probabilities.items():
                    new_token = token[cur_token_len:]
                    if not new_token:
                        new_probabilities = {}
                        break
                    new_prob = prob * scaling_factor
                    new_probabilities[new_token] = new_prob
                cur_token_dict[model_idx] = new_probabilities
        else:
            cur_token_dict = {model_idx: [] for model_idx in range(len(models))}
                
    return generated_text

def extract(pred_text):
    try:
        pred_text = pred_text[pred_text.find('answer is'):]
        pred_text = str(pred_text)
        left_index = pred_text.find('{')
        right_index = pred_text.find('}')
        raw_pred_answer = pred_text[left_index + 1:right_index]
        return raw_pred_answer
    except:
        return ''

if __name__ == "__main__":
    cor = 0
    with open(test_file, mode='r', encoding='utf-8') as f1, open(result_file_path, 'w', encoding='utf-8') as f:
        data = json.load(f1)
        idx = 0
        for datum in data:
            print("sample:", idx)
            f.write(f"------------------idx:{idx}--------------------")
            input_str = datum['input']
            prompt = example.format(input=input_str)
            print(prompt)
            reference = datum['output']
    
            response = collaborative_generate(prompt, max_length).replace(prompt, '')
            print("label:", reference)
            f.write("label:" + reference + '\n')
            extracted_answer = extract(response)
            if extracted_answer == reference:
                print('True')
                f.write('Judge:'+'True'+'\n')
                cor += 1
            else: 
                print('False')
                f.write('Judge:'+'False'+'\n')
            acc = cor / (idx+1)
            
            print("acc:", acc)
            print('\n')
            f.write("acc:" + str(acc) +'\n\n')
            idx += 1
            f.flush()
