import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple, Optional, Iterator
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse
import time
import os

from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus

def load_prompts_from_jsonl(input_path: str) -> List[Dict]:
    """Load prompts (with metadata) from a JSONL file."""
    with open(input_path, 'r') as f:
        return [json.loads(line) for line in f]

class Generator:
    def __init__(
        self,
        model_path: str,
        num_generations: int = 64,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
    ):
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.num_generations = num_generations
        
        self.sampling_params = SamplingParams(
            n=self.num_generations,
            temperature=0.8,
            top_p=0.95,
            max_tokens=4096,
        )

    def generate_responses(self, prompt: str) -> List[str]:
        """Generate multiple responses for a given prompt."""
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        outputs = self.model.generate(
            prompts=[chat_prompt],
            sampling_params=self.sampling_params
        )
        return [output.text for output in outputs[0].outputs]

    def process_prompt(self, prompt_row: Dict) -> Dict:
        """
        Process a single prompt (with metadata) through generation.
        The prompt_row should include at least a 'prompt' key.
        """
        prompt_text = prompt_row.get("prompt", "")
        responses = self.generate_responses(prompt_text)
        result = prompt_row.copy()
        result["responses"] = responses
        return result

    def generate_responses_stream(self, prompt_rows: Iterator[Dict], output_path: str):
        """Generate responses for a stream of prompts (with metadata) in local file mode."""
        with open(output_path, 'w') as f:
            for prompt_row in tqdm(prompt_rows):
                result = self.process_prompt(prompt_row)
                f.write(json.dumps(result) + '\n')
                f.flush()

class Scorer:
    # TODO port to VLLM
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device

    def score_responses(self, prompt: str, responses: List[str]) -> List[float]:
        """Score all responses in a single batch, handling sequences > 4096 tokens."""
        batch_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            for response in responses
        ]
    
        # Encode without padding to check lengths.
        encoded_inputs = [
            self.tokenizer.apply_chat_template(
                message,
                return_tensors="pt",
                padding=False
            )
            for message in batch_messages
        ]
    
        sequence_lengths = [len(x[0]) for x in encoded_inputs]
        valid_indices = [i for i, length in enumerate(sequence_lengths) if length <= 4096]
        excluded_indices = [i for i, length in enumerate(sequence_lengths) if length > 4096]

        invalid_objs = {
            "reward": -1,
            "coeff": -1,
            "score": -1,
        }
    
        if not valid_indices:
            # TODO the format on the second return value is not consistent.
            return [-1.0] * len(responses), [{}] * len(responses)
    
        valid_messages = [batch_messages[i] for i in valid_indices]
        batch_inputs = self.tokenizer.apply_chat_template(
            valid_messages,
            return_tensors="pt",
            padding=True
        ).to(self.device)
    
        with torch.no_grad():
            outputs = self.model(batch_inputs)
            valid_scores = outputs.score.cpu().float().tolist()

            multi_obj_rewards = outputs.rewards.cpu().float().tolist()

            gating_output = outputs.gating_output
            obj_transform = self.model.reward_transform_matrix.data
            multi_obj_coeffs = gating_output @ obj_transform.T
            multi_obj_coeffs = multi_obj_coeffs.cpu().float().tolist()

            attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence', 'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score', 'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness', 'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe', 'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity', 'code-style','code-explanation','code-instruction-following','code-readability']

        all_scores = []
        all_objs = []
        valid_idx = 0
        for i in range(len(responses)):
            objective_scores = {}
            score = -1
            if i in excluded_indices:
                all_scores.append(-1.0)
                for _, attribute in enumerate(attributes):
                    objective_scores[attribute] = {
                        "reward": -1,
                        "coeff": -1,
                        "score": -1,
                    }
                all_objs.append(objective_scores)
            else:
                score = valid_scores[valid_idx]
                for j, attribute in enumerate(attributes):
                    objective_scores[attribute] = {
                        "reward": multi_obj_rewards[valid_idx][j],
                        "coeff": multi_obj_coeffs[valid_idx][j],
                        "score_contribution": multi_obj_rewards[valid_idx][j] * multi_obj_coeffs[valid_idx][j],
                    }
                valid_idx += 1
            all_scores.append(score)
            all_objs.append(objective_scores)
    
        return all_scores, all_objs

    def score_generation_output(self, generation_result: Dict) -> Dict:
        scores, objs = self.score_responses(generation_result['prompt'], generation_result['responses'])
        
        scored_responses = [
            {'response': response, 'score': score, "objectives": obj}
            for response, score, obj in zip(generation_result['responses'], scores, objs)
        ]
        sorted_responses = sorted(scored_responses, key=lambda x: x['score'], reverse=True)
        best_response = sorted_responses[0]
        worst_response = sorted_responses[-1]
        result = {
            'prompt': generation_result['prompt'],
            'best_response': best_response,
            'worst_response': worst_response,
            'rip_metrics': {
                'rejected_response_reward': worst_response['score'],
                'rejected_response_length': len(worst_response['response']),
                'reward_gap': best_response['score'] - worst_response['score']
            },
            'all_responses': scored_responses,
        }
        return result

    

def apply_rip_filtering(
    input_path: str,
    accepted_path: str,
    rejected_path: str,
    rejected_reward_threshold: float = 0.5,
    rejected_length_threshold: float = 0.5,
    reward_gap_threshold: float = 0.5,
) -> Tuple[int, int]:
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    df = pd.DataFrame(results)
    metrics_df = pd.DataFrame([r['rip_metrics'] for r in results])
    
    rej_reward_val = metrics_df['rejected_response_reward'].quantile(rejected_reward_threshold)
    rej_length_val = metrics_df['rejected_response_length'].quantile(rejected_length_threshold)
    reward_gap_val = metrics_df['reward_gap'].quantile(reward_gap_threshold)
    
    mask = (
        (metrics_df['rejected_response_reward'] >= rej_reward_val) &
        (metrics_df['rejected_response_length'] >= rej_length_val) &
        (metrics_df['reward_gap'] <= reward_gap_val)
    )
    
    accepted = df[mask]
    rejected = df[~mask]
    
    with open(accepted_path, 'w') as f:
        for record in accepted.to_dict('records'):
            f.write(json.dumps(record) + '\n')
            
    with open(rejected_path, 'w') as f:
        for record in rejected.to_dict('records'):
            f.write(json.dumps(record) + '\n')
    
    return len(accepted), len(rejected)


def get_work(dispatcher_server):
    client = WorkClient(dispatcher_server)
    print(f"Using dispatcher server at {dispatcher_server}")
    while True:
        resp = client.get_work(batch_size=1)
        if resp.status == WorkStatus.ALL_WORK_COMPLETE:
            print("All work complete. Exiting.")
            break
        elif resp.status == WorkStatus.RETRY:
            print(f"No work available; retry in {resp.retry_in} seconds.")
            time.sleep(resp.retry_in)
            continue
        elif resp.status == WorkStatus.SERVER_UNAVAILABLE:
            print("Server is unavailable. Exiting.")
            break
        elif resp.status == WorkStatus.OK:
            batch = resp.items
            for work in batch:
                yield work
            client.submit_results(batch)
        else:
            print("Unexpected status from server; exiting.")
            break

def main():
    parser = argparse.ArgumentParser(description='RIP Tool')
    parser.add_argument('mode', choices=['generate', 'score', 'filter'], 
                        help='Operation mode')
    
    # vLLM specific arguments
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help='Maximum model context length')
    

    # Generation and scoring argument.
    parser.add_argument('--dispatcher_server', type=str, default=None,
                        help='If provided (in host:port format), generate mode will use the dispatcher server')
    
    # Generation arguments
    parser.add_argument('--base_model_path', type=str,
                        help='Path to the base model for generation')
    parser.add_argument('--num_generations', type=int, default=64,
                        help='Number of generations per prompt')
    
    # Scoring arguments
    parser.add_argument('--reward_model_path', type=str,
                        help='Path to the reward model')
    
    # Filtering arguments
    parser.add_argument('--input_path', type=str,
                        help='Input path for prompts (filter mode)')
    parser.add_argument('--accepted_path', type=str,
                        help='Output path for accepted prompts (filter mode)')
    parser.add_argument('--rejected_path', type=str,
                        help='Output path for rejected prompts (filter mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        if not (args.base_model_path and args.dispatcher_server):
            raise ValueError("--base_model_path and --dispatcher_server required for generate mode")
        generator = Generator(
            model_path=args.base_model_path,
            num_generations=args.num_generations,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )

        for work in get_work(args.dispatcher_server):
            row = json.loads(work.content)
            #{"messages": [{"role": "user", "content": "..."}, ...]
            result = generator.process_prompt({"prompt": row["messages"][0]["content"]})
            work.set_result(json.dumps(result))

    elif args.mode == 'score':
        if not (args.reward_model_path and args.dispatcher_server):
            raise ValueError("--reward_model_path required and --dispatcher_server required for score mode")
        scorer = Scorer(model_path=args.reward_model_path)

        for work in get_work(args.dispatcher_server):
            row = json.loads(work.content)
            result = scorer.score_generation_output(row)
            work.set_result(json.dumps(result))

    elif args.mode == 'filter':
        if not (args.input_path and args.accepted_path and args.rejected_path):
            raise ValueError("--input_path, --accepted_path, and --rejected_path required for filter mode")
        num_accepted, num_rejected = apply_rip_filtering(
            args.input_path,
            args.accepted_path,
            args.rejected_path
        )
        print("Filtering complete:")
        print(f"Accepted prompts: {num_accepted}")
        print(f"Rejected prompts: {num_rejected}")

if __name__ == "__main__":
    main()
