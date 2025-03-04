import re
import json
from copy import deepcopy
from datasets import load_dataset, Dataset, concatenate_datasets

datasets = []
for class_name in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
    ds = load_dataset("EleutherAI/hendrycks_math", class_name)['train']
    datasets.append(ds)
ds = concatenate_datasets(datasets)

for level in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:

    selected_levels = [level]

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

    # ds[0].keys()
    # dict_keys(['problem', 'level', 'type', 'solution'])

    with open("/shared/nas2/xiusic/wangyu/output/responses.json", "r") as file:
        responses = json.load(file)

    # responses = responses[:len(responses) // 10]

    context_messages = []
    labels = []
    levels = []

    for idx, response in enumerate(responses):
        # use re to retrieve the answer in "\box{}"
        # if not "\\boxed" in response['generated_text']:
        if not "\\boxed" in response:
            continue

        if not ds[idx % len(ds)]['level'] in selected_levels:
            continue

        # answer = re.search(r'\\boxed{([^}]*)}', response['generated_text']).group(1)
        answer = re.search(r'\\boxed{([^}]*)}', response).group(1)
        try:
            ground_truth = re.search(r'\\boxed{([^}]*)}', ds[idx % len(ds)]['solution']).group(1)
        except:
            continue

        problem = ds[idx % len(ds)]['problem']

        cur_prompt = deepcopy(prompt_template)
        # cur_prompt[1]['content'] = cur_prompt[1]['content'].format(problem=problem, solution=response['generated_text'])
        cur_prompt[1]['content'] = cur_prompt[1]['content'].format(problem=problem, solution=response)

        context_messages.append(cur_prompt)
        labels.append("yes" if answer == ground_truth else "no")
        levels.append(ds[idx % len(ds)]['level'])

    print("Number of correct responses:", sum([x=='yes' for x in labels]))
    print("Number of all responses:", len(labels))

    for unique_level in set(levels):
        print(f"Level {unique_level} has {levels.count(unique_level)} responses")
        print(f"level {unique_level} accuracy:", sum([labels[i] == 'yes' for i in range(len(labels)) if levels[i] == unique_level]) / levels.count(unique_level))

    for idx, item in enumerate(ds):
        if not item['level'] in selected_levels:
            continue
        solution = item['solution']
        cur_prompt = deepcopy(prompt_template)
        cur_prompt[1]['content'] = cur_prompt[1]['content'].format(problem=item['problem'], solution=solution)
        try:
            ground_truth = re.search(r'\\boxed{([^}]*)}', solution).group(1)
        except:
            continue
        context_messages.append(cur_prompt)
        labels.append("yes")

    print("Number of responses:", len(labels))

    dataset = Dataset.from_dict({
            'context_messages': context_messages,
            'labels': labels
    })

    dataset.save_to_disk("/shared/nas2/xiusic/wangyu/data/math_data_level" + level.split(" ")[1])
