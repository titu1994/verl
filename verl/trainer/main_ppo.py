# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import os
import ray
import json
import hydra
from collections import defaultdict

def processed_code_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # ground truth is reward_model key, correct_list is part of it
    import re
    # from nemo_skills.code_execution.math_grader import extract_answer

    solution_strs = solution_str
    ground_truths = ground_truth

    results = []
    metrics = {'pass@k': defaultdict(list), 'pass@1': []}
    for i, (solution_str, ground_truth) in enumerate(zip(solution_strs, ground_truths)):
        non_tensor_datum = json.loads(ground_truth)
        
        if non_tensor_datum['problemtype'] == 'math':
            assert False, 'Math problem type is not supported'
            pred = extract_answer(solution_str)
        else:
            if 'prediction' in non_tensor_datum:
                prediction = non_tensor_datum['prediction']
                assert type(prediction) == list, type(prediction)
                pred = '\n'.join([k.strip() for k in prediction])
            else:
                code_block = re.findall(r'```(.*?)```', solution_str, re.DOTALL)
                if not code_block:
                    pred = solution_str
                else:
                    pred = code_block[-1]

        correct_list = non_tensor_datum['correct_list']

        correct = all(correct_list)
        acc = float(correct)
        reward = 1.0 if correct else 0.0

        if extra_info and 'uid' in extra_info[i]:
            uid = extra_info[i]['uid']
            metrics['pass@k'][uid].append(correct)
        metrics['pass@1'].append(acc)

        results.append({
            'score': reward,
            'acc': acc,
            'pred': pred,
        })

    n_prompts = len(metrics['pass@k'])
    if n_prompts > 0:
        metrics['reward/all_pass'] = sum(all(metrics['pass@k'][uid]) for uid in metrics['pass@k']) / n_prompts
        metrics['reward/pass@k'] = sum(any(metrics['pass@k'][uid]) for uid in metrics['pass@k']) / n_prompts
        metrics['reward/pass@1'] = sum(metrics['pass@1']) / len(metrics['pass@1'])
    del metrics['pass@k']
    del metrics['pass@1']

    return {
        'metrics': metrics,
        'results': results,
    }


def sandbox_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from nemo_skills.code_execution.sandbox import LocalSandbox
    from nemo_skills.code_execution.math_grader import extract_answer
    sandbox = LocalSandbox()
    pred = extract_answer(solution_str)
    correct = sandbox.is_output_correct(pred, ground_truth)
    acc = float(correct)
    reward = 1.0 if correct else -1.0
    return {
        'score': reward,
        'acc': acc,
        'pred': pred,
    }

def judge_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from nemo_skills.training.openrlhf.math_reward import reward_func
    from nemo_skills.code_execution.math_grader import extract_answer
    prompt_metadata = []
    for gt, extra, in zip(ground_truth, extra_info):
        prompt_metadata.append({
            "problem": extra['problem'],
            "expected_answer": gt,
        })
    correct = reward_func(solution_str, None, prompt_metadata)

    results = []
    for i, (acc, solution) in enumerate(zip(correct, solution_str)):
        reward = 1.0 if acc else -1.0
        pred = extract_answer(solution)
        result = {
            "score": reward,
            "acc": acc.item(),
            "pred": pred,
        }
        results.append(result)

    return results


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == 'batched':
            from verl.workers.reward_manager import BatchRewardManager
            reward_manager_cls = BatchRewardManager
        elif reward_manager_name == 'code_sandbox_reward':
            from verl.utils.reward_score.nemo_code.code_sandbox_reward import RewardManager as CodeSandboxRewardManager
            reward_manager_cls = CodeSandboxRewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                    num_examine=0,
                                    compute_score=compute_score,
                                    reward_fn_key=config.data.reward_fn_key,
                                    max_resp_len=config.data.max_response_length,
                                    overlong_buffer_cfg=config.custom_reward_function.overlong_buffer)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                        num_examine=0,
                                        compute_score=compute_score,
                                        reward_fn_key=config.data.reward_fn_key,
                                        max_resp_len=config.data.max_response_length,
                                        overlong_buffer_cfg=config.custom_reward_function.overlong_buffer)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
