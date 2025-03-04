import re
import json
from copy import deepcopy
from datasets import load_dataset, Dataset, concatenate_datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--prompt", type=str, default='new')
args = parser.parse_args()


datasets = []
for class_name in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
    ds = load_dataset("EleutherAI/hendrycks_math", class_name)['test']
    datasets.append(ds)
ds = concatenate_datasets(datasets)

selected_levels = ["Level 1", "Level 2", "Level 3", "Level 4", 'Level 5']

if args.prompt == 'new':
    prompt_template = [
        {
            'role': 'system',
            'content': ("You are a mathematical expert and are able to evaluate the given solution."
                        "Conduct thorough evaluations on every step in the solution before making any prediction. Do not give your judge until the evaluation is finished. Put your judge in \\box{} "
                        "For instance, your response should be: [Evaluation Details]\n\n\\box{yes/no}")
        },
        {
            'role': 'user',
            'content': "The problem is\n\n{problem}\n\nThe generated solution is\n\n{solution}\n\nPlease evaluate the solution and give your final judge."
        }
    ]
elif args.prompt == 'old':
    prompt_template = [
        {
            'role': 'system',
            'content': ("You are a mathematical expert and are able to evaluate whether the given solution to a problem is correct. "
                        "Give detailed explanations for your evaluation and put your final judge yes/no in \\box{} "
                        "For instance, your response should be: [some evaluations here]\n\n\\box{yes/no}")
        },
        {
            'role': 'user',
            'content': "The problem is\n\n{problem}\n\nThe generated solution is\n\n{solution}\n\nPlease evaluate the solution and give your final judge."
        }
    ]
else:
    raise ValueError("Invalid prompt type")

# load responses_test:
with open("/shared/nas2/xiusic/wangyu/output/responses_test.json", "r") as file:
    responses = json.load(file)

# load model:
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
llm = LLM(args.model)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

prompts = []
labels = []
for idx, response in enumerate(responses):
    template_copy = deepcopy(prompt_template)
    template_copy[1]['content'] = template_copy[1]['content'].format(problem=ds[idx]['problem'], solution=response)
    prompt = tok.apply_chat_template(template_copy, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)

    if not "\\boxed" in response:
        labels.append(-1)
        continue
    answer = re.search(r'\\boxed{([^}]*)}', response).group(1)

    try:
        ground_truth = re.search(r'\\boxed{([^}]*)}', ds[idx % len(ds)]['solution']).group(1)
    except:
        labels.append(-1)
        continue
    
    if answer == ground_truth:
        labels.append(1)
    else:
        labels.append(0)

prompts = prompts[:1]
labels = labels[:1]

outputs = llm.generate(prompts, sampling_params=sampling_params)

predictions = []
for output in outputs:
    if "{yes}" in output.outputs[0].text.split("\n")[-1]:
        predictions.append("yes")
    elif "{no}" in output.outputs[0].text.split("\n")[-1]:
        predictions.append("no")
    else:
        predictions.append("error")

correct = 0
total = 0
error = 0

for label, pred in zip(labels, predictions):
    if pred == 'error':
        error += 1
    if label == -1:
        continue
    if label == 1 and pred == 'yes':
        correct += 1
    elif label == 0 and pred == 'no':
        correct += 1
    total += 1

print("Accuracy:", correct / total)
print("Error Rate:", error / len(labels))
