import argparse
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Combine sample_response_test and sample_response functionalities with argparse.")
    parser.add_argument(
        "--split", 
        default="test", 
        choices=["train", "test"], 
        help="Which dataset split to use: train or test."
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="Number of times to repeat the dataset."
    )
    parser.add_argument(
        "--template-type", 
        default="problem_statement", 
        choices=["problem_statement", "final_answer"],
        help="Which prompt template to use: 'problem_statement' or 'final_answer'."
    )
    parser.add_argument(
        "--outfile",
        default="responses.json",
        help="File name for saving the generated responses."
    )
    parser.add_argument(
        "--max-tokens",
        default=4096,
        type=int,
        help="Max tokens for generation."
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--top-p",
        default=0.95,
        type=float,
        help="Top-p for sampling."
    )

    args = parser.parse_args()

    # Load the model, tokenizer, and set sampling params
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens
    )

    system_prompt = "You are a professional mathematician. Please solve the following problem step by step and provide a final answer enclosed in \\box{}"

    if args.template_style == 'openai':
        template = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()
    else:
        template = """**Problem Statement**:
{problem}

**Solution:**"""

    def make_user_prompt(item):
        return template.format(problem=item['problem'])
    
    # Build prompts
    prompts = []
    for class_name in ['algebra', 'counting_and_probability', 'geometry', 
                       'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
        ds = load_dataset("EleutherAI/hendrycks_math", class_name)[args.split]
        for item in ds:
            # Construct the prompt with role-based chat format
            prompts.append(
                tok.apply_chat_template(
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': make_user_prompt(item)}
                    ],
                    tokenize=False, 
                    add_generation_prompt=True
                )
            )

    if args.repeat > 1:
        prompts = prompts * args.repeat

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Collect responses
    all_responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        all_responses.append(generated_text)

    # Save to JSON
    with open(args.outfile, "w") as file:
        json.dump(all_responses, file, indent=2)

    print(f"Saved {len(all_responses)} responses to {args.outfile}")

if __name__ == "__main__":
    main()
