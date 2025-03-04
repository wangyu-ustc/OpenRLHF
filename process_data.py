import copy
from datasets import load_dataset, concatenate_datasets, Dataset

prompt_template = [
    {
        'role': 'system',
        # 'content': (
        #     "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
        #     "to the user question displayed below. "
        #     "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
        #     "Then, compare the following two conversations between the user and the AI assistants, and provide your evaluation "
        #     "according to the rubric items. Ensure that the order in which the responses are presented does not influence your decision, and do not "
        #     "let response length or assistant names affect your evaluation. Be as objective as possible. "
        #     "Enclose your complete evaluation explanation and final verdict within <eval> and </eval> tags. "
        #     "After providing your explanation, "
        #     "output your final verdict by strictly following this format: '[[A]]' if assistant A is better, '[[B]]' if assistant B is better, '[[Tie]]' if they are equally good."
        #     "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here according to the rubric items</eval>\n\n<answer>[[A/B/Tie]]</answer>"
        # )
        'content': (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants "
            "to the user question displayed below. "
            "Begin your evaluation by first generating the rubric items. Enclose this section within <rubric> and </rubric> tags. "
            "Then, compare the following two conversations between the user and the AI assistants, and provide your evaluation "
            "according to the rubric items. Ensure that the order in which the responses are presented does not influence your decision, and do not "
            "let response length or assistant names affect your evaluation. Be as objective as possible. "
            "Enclose your complete evaluation explanation and final verdict within <eval> and </eval> tags. "
            "After providing your explanation, "
            "output your final verdict by strictly following this format: '[[A]]' if assistant A is better, '[[B]]' if assistant B is better."
            "i.e., <rubric>rubric items here</rubric>\n\n<eval>detailed evaluation here according to the rubric items</eval>\n\n<answer>[[A/B]]</answer>"
        )
    },
    {
        'role': 'user',
        'content': (
            "[The following is the conversation between Assistant A and the user]\n{conversation1}\n\n"
            "[The following is the conversation between Assistant B and the user]\n{conversation2}"
        )
    }
]


def collect_chat_data():

    def reformat_conversation(conv, assistant_name):
        conversations = []
        for item in conv:
            if item['role'] == 'user':
                conversations.append('User' + ": " + item['content'])
            else:
                conversations.append(assistant_name + ": " + item['content'])
        return '\n'.join(conversations)

    def process_data(item):
        cur_prompt = copy.deepcopy(prompt_template)
        cur_prompt[1]['content'] = cur_prompt[1]['content'].format(
            conversation1=reformat_conversation(item['conversation_a'], 'Assistant A'),
            conversation2=reformat_conversation(item['conversation_b'], 'Assistant B')
        )
        item['context_messages'] = cur_prompt
        return item

    ds = load_dataset("lmsys/mt_bench_human_judgments")['gpt4_pair']
    # ds2 = load_dataset("lmsys/chatbot_arena_conversations")['train']

    ds = ds.map(process_data)
    # ds2 = ds2.map(process_data)

    # Define the keys (columns) to keep
    desired_keys = ["context_messages", "winner"]  # Modify this list based on your needs
    ds = ds.select_columns(desired_keys)
    # ds2 = ds2.select_columns(desired_keys)
    # ds = concatenate_datasets([ds, ds2])

    return ds

def collect_helpfulness():

    ds = load_dataset("nvidia/HelpSteer2")['train']

    context_messages = []
    winner = []

    conversations = []
    rating = []

    for idx, item in enumerate(ds):

        if idx % 2 == 0:
            conversations.append('User: ' + item['prompt'] + '\n' + "Assistant A: " + item['response'])
            rating.append(item['helpfulness'])
            continue
        else:
            conversations.append('User: ' + item['prompt'] + '\n' + "Assistant B: " + item['response'])
            rating.append(item['helpfulness'])

            cur_prompt = copy.deepcopy(prompt_template)
            cur_prompt[1]['content'] = cur_prompt[1]['content'].format(
                conversation1=conversations[0],
                conversation2=conversations[1]
            )

            context_messages.append(cur_prompt)

            if rating[0] > rating[1]:
                winner.append('model_a')
            else:
                winner.append('model_b')
            
            conversations = []
            rating = []

    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset

def collect_safe():

    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")['train']
    context_messages = []
    winner = []

    same_better_and_safer_count = 0

    for item in ds:
        conversation1 = 'User: ' + item['prompt'] + '\n' + "Assistant A: " + item['response_0']
        conversation2 = 'User: ' + item['prompt'] + '\n' + "Assistant B: " + item['response_1']
        cur_prompt = copy.deepcopy(prompt_template)
        cur_prompt[1]['content'] = cur_prompt[1]['content'].format(
            conversation1=conversation1,
            conversation2=conversation2
        )
        context_messages.append(cur_prompt)
        if item['safer_response_id'] == 0:
            winner.append('model_a')
        else:
            winner.append('model_b')
        
        if item['safer_response_id'] == item['better_response_id']:
            same_better_and_safer_count += 1
    
    print("consistent ratio:", same_better_and_safer_count / len(ds))

    dataset = Dataset.from_dict({
        'context_messages': context_messages,
        'winner': winner
    })

    return dataset

if __name__ == '__main__':
    
    ds_safe = collect_safe()
    ds_helpfulness = collect_helpfulness()
    ds_chat = collect_chat_data()

    # print(len(ds_safe))
    # print(len(ds_helpfulness))
    print(len(ds_chat))

    ds = concatenate_datasets([ds_helpfulness, ds_chat, ds_safe])
    # ds_chat.save_to_disk('/shared/nas2/xiusic/wangyu/data/chat_data')

    # extract data with "winner" == "model_a" or "winner" == "model_b"
    ds = ds.filter(lambda x: x['winner'] == 'model_a' or x['winner'] == 'model_b')

    ds.save_to_disk('/shared/nas2/xiusic/wangyu/data/merged_data')
    