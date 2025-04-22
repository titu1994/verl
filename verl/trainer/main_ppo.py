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
from collections import defaultdict
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import ray
import hydra

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


class BatchedRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, overlong_buffer_cfg=None, max_response_length=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_response_length = max_response_length

        if self.overlong_buffer_cfg is not None:
            assert self.max_response_length is not None, f'max resp length must be provided if {self.overlong_buffer_cfg=}, but got None'
            assert self.overlong_buffer_cfg.enable in [True, False], f'{self.overlong_buffer_cfg.enable=} must be a boolean'
            assert self.overlong_buffer_cfg.len is not None, f'{self.overlong_buffer_cfg.len=} must be provided'
            assert self.overlong_buffer_cfg.penalty_factor is not None, f'{self.overlong_buffer_cfg.penalty_factor=} must be provided'


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        data_sources = []
        solutions = []
        ground_truths = []
        extra_infos = []
        valid_response_lengths = []
        prompts = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            data_sources.append(data_source)
            solutions.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)
            valid_response_lengths.append(valid_response_length)
            prompts.append(prompt_str)

        result = self.compute_score(
            data_sources=data_sources,
            solution_strs=solutions,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
        scores = []
        if isinstance(result, dict):
            # result is a dictionary with "score", and possibly other arrays like "acc", "pred", etc.
            score = result['score']
            scores = score
            for key, value in result.items():
                for v in value:
                    reward_extra_info[key].append(v)
        else:
            # result is just a list/array of scores
            scores = result

        # Make sure the number of scores we got matches the batch size
        assert len(scores) == reward_tensor.shape[0], (
            f'{len(scores)=} != {reward_tensor.shape[0]=}, number of scores does not match the number of data'
        )

        for i in range(len(data)):
            data_source = data_sources[i]
            valid_response_length = valid_response_lengths[i]

            final_score = scores[i]
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_response_length - overlong_buffer_len
                exceeded_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceeded_len / overlong_buffer_len * overlong_penalty_factor, 0)
                final_score += overlong_reward  # now final_score is the *penalized* score
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = final_score

            if "score" in reward_extra_info:
                reward_extra_info["score"][i] = final_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]",  prompts[i])
                print("[response]", solutions[i])
                print("[ground_truth]", ground_truths[i])

                if 'pred' in reward_extra_info.keys():
                    print("[pred]", reward_extra_info['pred'][i])

                print('[score]', reward_tensor[i, valid_response_length - 1])
                print("[data_source]", data_source)
        if return_dict:
            return {'reward_tensor': reward_tensor, 'reward_extra_info': reward_extra_info}
        return reward_tensor

def judge_compute_score(data_sources, solution_strs, ground_truths, extra_infos=None):
    from nemo_skills.training.openrlhf.math_reward import reward_func
    prompt_metadata = []
    for ground_truth, extra_info, in zip(ground_truths, extra_infos):
        prompt_metadata.append({
            "problem": extra_info['problem'],
            "expected_answer": ground_truth,
        })
    return reward_func(solution_strs, None, prompt_metadata)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    remote_rm_url = config.reward_model.get('reward_manager', None)
    compute_score = config.reward_model.get('compute_score', None)
    compute_score_fn = None
    if (remote_rm_url is not None) and (remote_rm_url.endswith('.py')):
        print(f"Loading custom `reward_func(queries, prompts)` from {remote_rm_url}")
        import importlib.util
        import inspect

        spec = importlib.util.spec_from_file_location("reward_manager_module", remote_rm_url)
        reward_module = importlib.util.module_from_spec(spec)
        sys.modules['reward_manager_module'] = reward_module
        spec.loader.exec_module(reward_module)
        reward_manager = reward_module.RewardManager
    elif remote_rm_url == 'code_sandbox_reward':
        from verl.utils.reward_score.custom.code_sandbox_reward import RewardManager
        reward_manager = RewardManager
    elif compute_score == 'math-judge':
        compute_score_fn = judge_compute_score
    else:
        reward_manager = None
        compute_score_fn = None
    run_ppo(config, reward_manager=reward_manager, compute_score=compute_score_fn)


def run_ppo(config, reward_manager=None, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, reward_manager=reward_manager, compute_score=compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, reward_manager=None, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

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
    if reward_manager is not None:
        reward_manager_cls = reward_manager
    elif reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == 'batched':
        reward_manager_cls = BatchedRewardManager
    else:
        raise NotImplementedError
    
    num_examine = config.reward_model.reward_manager.get('num_examine', 0)
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=num_examine, compute_score=compute_score, overlong_buffer_cfg=config.reward_model.reward_manager.get('overlong_buffer', None), max_response_length=config.data.max_response_length)

    # Turn off num_examine, context length too long
    if config.trainer.get('run_validation', True):
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=num_examine, compute_score=compute_score, overlong_buffer_cfg=config.reward_model.reward_manager.get('overlong_buffer', None), max_response_length=config.data.max_response_length)
    else:
        val_reward_fn = None

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
