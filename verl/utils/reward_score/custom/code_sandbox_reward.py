#' Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
import os
import uuid
import json
import numpy as np
import bisect
import pandas as pd
import torch

import threading
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from nemo_codegen.code_execution.code_execution import sandbox_code_execution
from nemo_codegen.code_execution.sandbox import get_sandbox

from verl import DataProto
from verl.utils import torch_functional as torch_fn


############################################################
# Utility Functions
# (Mostly unchanged -- some references to sandbox are parameterized)
############################################################

def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split('\n')
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines)

def find_sublist(lst, sublst):
    assert type(lst) == list, f"lst must be a list, got {type(lst)}"
    assert type(sublst) == list, f"sublst must be a list, got {type(sublst)}"
    if not sublst:
        return 0
    n, m = len(lst), len(sublst)
    for i in range(n - m + 1):
        if lst[i:i + m] == sublst:
            return i
    return -1

def rfind_sublist(lst, sublst):
    assert type(lst) == list, f"lst must be a list, got {type(lst)}"
    assert type(sublst) == list, f"sublst must be a list, got {type(sublst)}"
    if not sublst:
        return len(lst)
    n, m = len(lst), len(sublst)
    for i in range(n - m, -1, -1):
        if lst[i:i + m] == sublst:
            return i
    return -1

def count_sublist(lst, sublst):
    assert type(lst) == list, f"lst must be a list, got {type(lst)}"
    assert type(sublst) == list, f"sublst must be a list, got {type(sublst)}"
    assert len(lst) > 0, lst
    assert len(sublst) > 0, sublst
    count = 0
    i = 0
    n, m = len(lst), len(sublst)
    while i <= n - m:
        if lst[i:i + m] == sublst:
            count += 1
            i += m
        else:
            i += 1
    return count

def compute_math_score(solution_str, ground_truth, box_strict=True) -> float:
    retval = 0.
    extra_length = 0
    try:
        string_in_last_boxed, extra_length, strict = boxed_only_string(solution_str, last=True, box_strict=box_strict)
        if string_in_last_boxed is not None:
            if strict:
                answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.
        else:
            retval = -1.
    except Exception as e:
        retval = -1.
        print(e)

    return retval, extra_length

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

def boxed_only_string(string, last=False, box_strict=True):
    if last:
        idx = string.rfind("\\boxed")
    else:
        idx = string.find("\\boxed")

    if "\\boxed " in string:
        # quick direct parse
        value =  string.split("\\boxed ")[-1].split("$")
        return "\\boxed " + value[0], len(value[1]) + 1, True

    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            if box_strict:
                return None, 0, True
            else:
                return string, 0, False

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None, 0, True
    else:
        retval = string[idx:right_brace_idx + 1], len(string)-right_brace_idx-1, True

    return retval

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return f"\\frac{{{a}}}{{{b}}}"
    except (AssertionError, ValueError):
        return string

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string

def process_answer_tags(
    tokenizer,
    response_token_ids,
    answer_tag: str,
):
    start_answer_idx = 0
    answer_len = len(response_token_ids)
    extra_length = 0

    formatting = {'answer_open_count': None, 'answer_close_count': None, 'answer_len': None}
    if answer_tag == "":
        return start_answer_idx, answer_len, formatting
    
    open_tag_token_ids = tokenizer.encode(f'<{answer_tag}>', add_special_tokens=False)
    close_tag_token_ids = tokenizer.encode(f'</{answer_tag}>', add_special_tokens=False)

    answer_open_count = count_sublist(response_token_ids, open_tag_token_ids)
    answer_close_count = count_sublist(response_token_ids, close_tag_token_ids)

    temp = tokenizer.decode(response_token_ids)
    assert answer_open_count == temp.count(f'<{answer_tag}>'), (
        f"Answer open count mismatch {answer_open_count} != {temp.count(f'<{answer_tag}>')}\n\n"
        f"text: {temp}\n\nopen_tag_token_ids: {open_tag_token_ids}\n\n"
        f"response_token_ids: {response_token_ids}"
    )

    formatting['answer_open_count'] = answer_open_count
    formatting['answer_close_count'] = answer_close_count

    if answer_open_count == 1:
        start_answer_idx = find_sublist(response_token_ids, open_tag_token_ids) + len(open_tag_token_ids)
        end_answer_idx = rfind_sublist(response_token_ids, close_tag_token_ids)
        if end_answer_idx == -1:
            end_answer_idx = len(response_token_ids)
        else:
            extra_length = len(response_token_ids) - (end_answer_idx + len(close_tag_token_ids))
        answer_len = end_answer_idx - start_answer_idx
        formatting['answer_len'] = answer_len

    return start_answer_idx, answer_len, formatting

def process_think_tags(
    tokenizer,
    response_token_ids,
    think_tag: str,
    max_tokens: int,
):
    num_response_tokens = len(response_token_ids)
    num_reasoning_tokens = 0
    num_non_reasoning_tokens = num_response_tokens

    temp = tokenizer.decode(response_token_ids)
    open_tag = f'<{think_tag}>'
    close_tag = f'</{think_tag}>'
    open_tag_token_ids = tokenizer.encode(open_tag, add_special_tokens=False)
    close_tag_token_ids = tokenizer.encode(close_tag, add_special_tokens=False)

    think_open_count = count_sublist(response_token_ids, open_tag_token_ids)
    think_close_count = count_sublist(response_token_ids, close_tag_token_ids)
    formatting = {'think_open_count': think_open_count, 'think_close_count': think_close_count}

    post_think_token_idx = 0
    if think_close_count > 0:
        close_tag_start_idx = rfind_sublist(response_token_ids, close_tag_token_ids)
        num_reasoning_tokens = close_tag_start_idx
        post_think_token_idx = close_tag_start_idx + len(close_tag_token_ids)
        num_non_reasoning_tokens = num_response_tokens - post_think_token_idx
        assert 0 <= num_non_reasoning_tokens <= max_tokens, (
            f"Number of non-reasoning tokens {num_non_reasoning_tokens} "
            f"is greater than max tokens {max_tokens}"
        )
    
    return post_think_token_idx, formatting, num_reasoning_tokens, num_non_reasoning_tokens

def pass_at_k(n, c, k):
    if (n-c) < k:
        return 1.0
    p = 1. - np.prod(1. - k/np.arange(n - c + 1, n + 1))
    return p

def remove_code_fence(language: str, solution: str) -> str:
    if language.lower() == "python":
        solution = re.sub(r'^(?:\n)?[Pp]ython\s*', '', solution)
    elif language.lower() in ("cpp", "c++"):
        solution = re.sub(r'^(?:\n)?[Cc]pp\s*', '', solution)
    else:
        solution = re.sub(r'.*?\n(.*?)', r'\1', solution, flags=re.DOTALL)
    solution += "\n"
    return solution

def execute_stdin_tests(
    solution: str,
    cc_tests,
    limit_tests: int,
    sandbox,
    timeout,
    language,
) -> list[bool]:
    correct_list = []
    test_inputs = []
    test_outputs = []
    for key in cc_tests:
        if isinstance(cc_tests[key], dict):
            test_inputs.extend(cc_tests[key].get("input", []))
            test_outputs.extend(cc_tests[key].get("output", []))
    test_inputs = test_inputs[:limit_tests]
    test_outputs = test_outputs[:limit_tests]
    assert len(test_inputs) > 0, cc_tests

    for inp, out in zip(test_inputs, test_outputs):
        inp_clean = inp.replace("\r\n", "\n").replace("\r", "\n")
        out_clean = out.replace("\r\n", "\n").replace("\r", "\n")

        result, _ = sandbox.execute_code(
            solution,
            std_input=inp_clean,
            timeout=timeout,
            language=language,
        )
        stdout = result['stdout'].replace("\r\n", "\n").replace("\r", "\n")
        correct = (stdout.strip() == out_clean.strip())
        correct_list.append(correct)

    return correct_list

def dco_factor(n, N):
    p = n / N
    return N*((1-p)**(N-1)) * p/(1-(1-p)**N)

def compute_reward(config, problem_type, all_formatting, all_correct_lists, all_num_reasoning_tokens, all_num_non_reasoning_tokens, event):
    n = len(all_correct_lists)
    assert len(all_num_reasoning_tokens) == n
    assert len(all_num_non_reasoning_tokens) == n
    assert n > 0

    reward_fn_args = config.reward_model.get('reward_fn_args', {})
    code_reward_args = reward_fn_args.get('code', {})
    code_policy = code_reward_args.get('verifier_reward', 'fractional')
    group_policy = code_reward_args.get('verifier_group_reward', 'pass@1')

    unused_reasoning_penalty = reward_fn_args['penalties']['unused_reasoning']
    overlong_buffer_cfg = config.reward_model.reward_manager.overlong_buffer
    overlong_len_threshold = overlong_buffer_cfg.len
    overlong_penalty_factor = overlong_buffer_cfg.penalty_factor
    max_gen_len = config.data.max_response_length

    group_rewards = [0.] * n
    all_penalties = []
    for i in range(n):
        penalty = 0.0
        if event != 'val':
            for k, v in all_formatting.items():
                factor = config.reward_model['reward_fn_args']['penalties'].get(k, 0.0)
                if v[i] is None:
                    continue
                if k in ['code_fence_missing', 'math_box_missing']:
                    if v[i]:
                        penalty += factor
                elif k in ['ends_with_stop_phrase']:
                    if not v[i]:
                        penalty += factor
                elif k in ['think_open_count']:
                    if v[i] > 0:
                        penalty += factor
                elif k in ['think_close_count', 'answer_open_count', 'answer_close_count']:
                    if v[i] != 1:
                        penalty += factor
                elif k in ['answer_len']:
                    if v[i] == 0:
                        penalty += factor
        all_penalties.append(penalty)

    if problem_type == 'math':
        # each item_result has correct_list => single item [0 or 1 or -1 => mapped to pass/fail].
        pass_results = [arr[0] for arr in all_correct_lists]
        rewards = [float(c) for c in pass_results]  # e.g. 0 or 1
        rewards = [c - p for c, p in zip(rewards, all_penalties)]
        pass_or_not = pass_results
    elif problem_type == 'code':
        pass_or_not = []
        for c in all_correct_lists:
            # c is list[bool], pass if all True
            passed = bool(np.all(c))
            pass_or_not.append(passed)

        if event != 'val':
            unused_reasoning = [
                (max_gen_len - (all_num_reasoning_tokens[i] + all_num_non_reasoning_tokens[i])) / max_gen_len
                for i in range(n)
            ]
            overthreshold_answer = [
                max(0, (all_num_reasoning_tokens[i] + all_num_non_reasoning_tokens[i] - overlong_len_threshold) / overlong_len_threshold)
                for i in range(n)
            ]
            overthreshold_answer_penalty = [overthreshold_answer[i] * overlong_penalty_factor for i in range(n)]
            unused_reasoning_penalty_vals = [unused_reasoning[i] * unused_reasoning_penalty for i in range(n)]
            all_penalties = [p + u + o for p, u, o in zip(all_penalties, unused_reasoning_penalty_vals, overthreshold_answer_penalty)]

        if (code_policy == 'pass@1') or (event == 'val'):
            pass_or_not_float = [float(x) for x in pass_or_not]
            rewards = [-p if p > 0 else c for c, p in zip(pass_or_not_float, all_penalties)]
        elif code_policy == 'fractional':
            fraction_correct = [float(np.mean(c)) for c in all_correct_lists]
            rewards = [-p if p > 0 else c for c, p in zip(fraction_correct, all_penalties)]
        else:
            raise ValueError(f"Unsupported code policy: {code_policy}")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    group_pass_count = sum(int(k) for k in pass_or_not)
    group_pass_or_not = (group_pass_count > 0)

    if group_policy == 'pass@n':
        group_rewards = [float(group_pass_or_not)] * n
    elif group_policy == 'dco':
        if group_pass_or_not:
            factor = dco_factor(group_pass_count, n)
            rewards = [factor * r if r > 0 else r for r in rewards]
    elif group_policy == 'pass@1':
        pass
    else:
        raise ValueError(f"Unsupported group policy: {group_policy}")

    rewards = [float(x) for x in rewards]

    all_specific_penalties = [{} for _ in range(n)]
    for k in all_formatting:
        assert len(all_formatting[k]) == n, f'{k}, {len(all_formatting[k])} vs {n}'
        for i in range(n):
            all_specific_penalties[i][k] = all_formatting[k][i]

    return rewards, group_rewards, pass_or_not, group_pass_or_not, all_specific_penalties

def convert_functional_tests_to_asserts(func_name, tests):
    unit_tests = []
    outputs = set()
    for k, v in tests.items():
        if not isinstance(v, dict):
            continue
        outputs.update(v['output'])

    boolean_output = False
    if outputs.issubset({'true', 'false'}):
        boolean_output = True

    for k, v in tests.items():
        if not isinstance(v, dict):
            continue
        inputs = v['input']
        outputs = v['output']
        for inp, out in zip(inputs, outputs):
            inp = inp.replace("\n", ",")
            if boolean_output:
                test = f"\nassert str({func_name}({inp}).lower()) == str({out})\n"
            else:
                test = f"\nassert str({func_name}({inp})) == str({out})\n"
            unit_tests.append(test)
    return unit_tests


def get_num_reasoning_tokens(response_token_ids, fence_start, answer_string, tokenizer, language):
    decoded = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in response_token_ids]
    lens = [len(tok) for tok in decoded]
    lens = np.cumsum(lens)
    s = ''.join(decoded)

    substr = fence_start + answer_string[:20]
    idx = s.rfind(substr)
    if idx == -1:
        substr = f'```{language}'
    idx = s.rfind(substr)
    if idx == -1:
        substr = '```'
    idx = s.rfind(substr)
    if idx == -1:
        return 0, len(response_token_ids)

    idx = bisect.bisect_left(lens, idx)
    num_reasoning_tokens = idx
    num_non_reasoning_tokens = len(response_token_ids) - num_reasoning_tokens
    return num_reasoning_tokens, num_non_reasoning_tokens


def execute_via_sandbox(
    solution,
    tests,
    limit,
    sandbox,
    timeout,
    language,
    fn_name=None,
):
    test_type = 'stdin' if fn_name is None else 'functional'
    if test_type == 'stdin':
        correct_list = execute_stdin_tests(
            solution, tests, limit, sandbox, timeout, language
        )
    else:
        assert fn_name is not None
        if 'class Solution' in solution:
            solution += f"\n\n{fn_name} = Solution().{fn_name}"
        unit_tests = convert_functional_tests_to_asserts(fn_name, tests)
        res = sandbox_code_execution(
            completion=solution,
            unit_tests=unit_tests,
            timeout=timeout,
            sandbox=sandbox
        )
        correct_list = res['correct_tests']
    return correct_list


def compute_single_item_score(
    response_token_ids: list[int],
    tokenizer,
    non_tensor_datum: dict,
    config: dict,
    event: str | None = None,
    sandbox=None,
    continuous=False,
):
    """
    Evaluate correctness for a single item, returning:
      {'correct_list': [...],
       'formatting': {...},
       'num_reasoning_tokens': int,
       'num_non_reasoning_tokens': int}
    """

    response_token_ids = response_token_ids.cpu().numpy().tolist()

    formatting = {}
    correct_list = []

    think_tag = non_tensor_datum['think_tag']
    answer_tag = non_tensor_datum['answer_tag']
    stop_phrases = non_tensor_datum['stop_phrases']
    max_tokens = config.data.max_response_length
    problem_type = non_tensor_datum['problemtype']

    write_content = []

    # 1) Reasoning split
    if think_tag == '':
        post_think_token_idx = 0
        formatting['think_open_count']  = None
        formatting['think_close_count'] = None

    else:
        post_think_token_idx, think_fmt, num_reasoning_tokens, num_non_reasoning_tokens = (
            process_think_tags(
                tokenizer,
                response_token_ids,
                think_tag,
                max_tokens,
            )
        )
        formatting.update(think_fmt)

    # 2) Stop-phrase
    formatting['ends_with_stop_phrase'] = any(
        response_token_ids[-len(p):] == p for p in stop_phrases
    )

    # 3) answer-tag
    resp_after_think = response_token_ids[post_think_token_idx:]
    ans_start, ans_len, ans_fmt = process_answer_tags(tokenizer, resp_after_think, answer_tag)
    formatting.update(ans_fmt)

    if ans_len > 0:
        resp_after_think = resp_after_think[ans_start : ans_start + ans_len]

    response_text = tokenizer.decode(resp_after_think, skip_special_tokens=False)

    # 4) correctness
    if problem_type == 'math':
        expected = non_tensor_datum.get('expected_answer')
        res, _ = compute_math_score(response_text, expected, box_strict=True)
        formatting['math_box_missing'] = (res == -1.0)
        correct_list.append(int(res == 1.0))

    else:  # code
        language = non_tensor_datum.get('language', 'python')
        # code_block = re.search(r'```(.*?)```', response_text, re.DOTALL)
        code_block = re.findall(r'```(.*?)```', response_text, re.DOTALL)

        if not code_block:
            solution = response_text
            formatting['code_fence_missing'] = True
        else:
            #solution = code_block.group(1)
            solution = code_block[-1]
            formatting['code_fence_missing'] = False
            num_reasoning_tokens, num_non_reasoning_tokens = get_num_reasoning_tokens(response_token_ids, '```', solution, tokenizer, language)

        code_cfg = config.reward_model.get('reward_fn_args', {}).get('code', {})
        tests_used = code_cfg.get('unit_tests_to_use', ['public', 'private', 'generated'])

        all_tests = {
            'public'   : non_tensor_datum['public_test_cases'],
            'private'  : non_tensor_datum['private_test_cases'],
            'generated': non_tensor_datum['generated_test_cases']
        }
        tests = {k: v for k, v in all_tests.items() if k in tests_used}

        solution = remove_code_fence(language, solution)
        num_max_tests = code_cfg.get('limit_tests', 10) if event != 'val' else 10000
        timeout = code_cfg.get('timeout', 1.0)
        test_type = non_tensor_datum['testtype']

        if sandbox is None:
            # fallback
            sandbox = get_sandbox('fast')

        fn_name = non_tensor_datum.get('metadata', {}).get('func_name', None)
        assert test_type == 'stdin' or fn_name is not None, "fn_name is required for non-stdin tests"
        assert fn_name is None or test_type == 'functional', "fn_name is only supported for functional tests"

        correct_list = execute_via_sandbox(
            solution, 
            tests, 
            num_max_tests, 
            sandbox, 
            timeout, 
            language, 
            fn_name=fn_name,
        )

    return {
        'correct_list': correct_list,
        'formatting': formatting,
        'num_reasoning_tokens': num_reasoning_tokens,
        'num_non_reasoning_tokens': num_non_reasoning_tokens,
    }


def compute_item_results(data, tokenizer, config, event):
    """
    Spawns tasks to compute item_results for each item in 'data',
    respecting concurrency_per_sandbox for each sandbox.
    Returns a list of item_result dicts in the same order as data.
    """

    item_results = [None] * len(data)

    continuous = config.reward_model.reward_fn_args.code.get('verifier_reward', 'fractional') == 'fractional'
    if event == 'val':
        continuous = False

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, item in enumerate(data):
            batch = item.batch
            prompt_len = batch['prompts'].shape[-1]
            valid_resp_len = batch['attention_mask'][prompt_len:].sum()
            valid_resp_ids = batch['responses'][:valid_resp_len]

            non_tensor_datum = item.non_tensor_batch
            if 'metadata' in non_tensor_datum:
                non_tensor_datum['metadata'] = json.loads(non_tensor_datum['metadata'])

            fut = executor.submit(
                compute_single_item_score,
                response_token_ids=valid_resp_ids,
                tokenizer=tokenizer,
                non_tensor_datum=non_tensor_datum,
                config=config,
                event=event,
                sandbox='internal',
                continuous=continuous,
            )
            futures.append((i, fut))

        for i, fut in futures:
            item_results[i] = fut.result()

    return item_results


############################################################
# Post-exec aggregator (unchanged)
############################################################

def compute_score_post_exec(item_results, problem_type, event, config):
    all_correct_lists = []
    all_num_reasoning_tokens = []
    all_num_non_reasoning_tokens = []
    all_formatting = defaultdict(list)

    for item_result in item_results:
        all_correct_lists.append(item_result['correct_list'])
        all_num_reasoning_tokens.append(item_result['num_reasoning_tokens'])
        all_num_non_reasoning_tokens.append(item_result['num_non_reasoning_tokens'])
        for k, v in item_result['formatting'].items():
            all_formatting[k].append(v)

    (
        rewards, 
        group_rewards, 
        pass_or_not, 
        group_pass_or_not, 
        all_specific_penalties
    ) = compute_reward(
        config=config,
        problem_type=problem_type,
        all_formatting=all_formatting,
        all_correct_lists=all_correct_lists,
        all_num_reasoning_tokens=all_num_reasoning_tokens,
        all_num_non_reasoning_tokens=all_num_non_reasoning_tokens,
        event=event,
    )

    # shape: item_results is same length as all lists
    return (
        rewards, 
        group_rewards, 
        pass_or_not, 
        group_pass_or_not, 
        all_num_non_reasoning_tokens, 
        all_specific_penalties, 
        all_formatting
    )


def _process_group(args):
    """
    Aggregates item_results at the group (prompt) level for final logging/metrics.
    """
    (
        uid,
        group_idxes,
        data,
        item_results,
        event,
        config,
        tokenizer,
    ) = args

    write_content = {'uid': uid, 'responses': []}
    group_items = [data[i] for i in group_idxes]
    prompt_ids = group_items[0].batch['prompts']
    prompt_len = prompt_ids.shape[-1]
    valid_prompt_len = group_items[0].batch['attention_mask'][:prompt_len].sum()
    prompt_str = tokenizer.decode(prompt_ids[-valid_prompt_len:])
    write_content['prompt'] = prompt_str

    response_lengths = []
    prompt_num_pad_tokens_loc = []
    response_num_pad_tokens_loc = []
    for item in group_items:
        resp_ids = item.batch['responses'].numpy().tolist()
        valid_resp_len = item.batch['attention_mask'][prompt_len:].sum()
        response_lengths.append(valid_resp_len)
        prompt_num_pad_tokens_loc.append(prompt_len - valid_prompt_len)
        response_num_pad_tokens_loc.append(len(resp_ids) - valid_resp_len)

        resp_str = tokenizer.decode(resp_ids[:valid_resp_len], skip_special_tokens=False)
        write_content['responses'].append({'response': resp_str})

    item_results_group = [item_results[i] for i in group_idxes]
    problem_type = group_items[0].non_tensor_batch['problemtype']

    (
        rewards,
        group_rewards,
        pass_or_not,
        group_pass_or_not,
        num_non_reasoning_tokens,
        all_penalties,
        all_formatting,
    ) = compute_score_post_exec(item_results_group, problem_type, event, config)

    for i, _ in enumerate(group_items):
        write_content['responses'][i]['reward'] = rewards[i]
        write_content['responses'][i]['pass_or_not'] = pass_or_not[i]
        write_content['responses'][i]['formatting'] = {
            k: v[i] for k, v in all_formatting.items()
        }
    write_content['group_pass_count'] = group_pass_or_not

    metrics_local = [{} for _ in range(len(pass_or_not))]
    for i in range(len(pass_or_not)):
        metrics_local[i]['rewards/group_pass_count'] = float(group_pass_or_not)
        metrics_local[i]['rewards/pass_count'] = int(pass_or_not[i])
        metrics_local[i]['rewards/response_lengths'] = response_lengths[i]
        metrics_local[i]['rewards/num_reasoning_tokens'] = response_lengths[i] - num_non_reasoning_tokens[i]
        metrics_local[i].update(all_penalties[i])

    return (
        uid,
        group_idxes,
        rewards,
        group_rewards,
        pass_or_not,
        num_non_reasoning_tokens,
        response_num_pad_tokens_loc,
        prompt_num_pad_tokens_loc,
        response_lengths,
        write_content,
        metrics_local,
    )


def expand_slurm_nodelist(nodelist):
    """
    Turn a SLURM_JOB_NODELIST string into a list of hostnames.
    Handles:
      - prefix[01-03,05] style ranges
      - simple comma‑separated entries
      - mixed lists like a[1-2],b,c[05]
    """
    # 1) split on commas not inside brackets
    tokens = []
    buf = ""
    depth = 0
    for c in nodelist:
        if c == "[":
            depth += 1
            buf += c
        elif c == "]":
            depth -= 1
            buf += c
        elif c == "," and depth == 0:
            tokens.append(buf)
            buf = ""
        else:
            buf += c
    if buf:
        tokens.append(buf)

    # 2) expand each token
    hosts = []
    for tok in tokens:
        m = re.match(r"^(.*?)\[(.*?)\]$", tok)
        if m:
            prefix, inside = m.groups()
            for part in inside.split(","):
                if "-" in part:
                    start, end = part.split("-")
                    width = max(len(start), len(end))
                    for num in range(int(start), int(end) + 1):
                        hosts.append(f"{prefix}{str(num).zfill(width)}")
                else:
                    hosts.append(f"{prefix}{part}")
        else:
            hosts.append(tok)
    return hosts


class RewardManager:
    """
    Accepts a list of hosts and a port at init, creates sandboxes.
    In __call__, we do:
      1. compute_item_results(...) [NEW method]
      2. aggregate rewards with _process_group
      3. optionally apply logprob shaping
      4. return reward_tensor or the full tuple
    """

    name = 'code_sandbox_reward'

    def __init__(
        self,
        tokenizer,
        num_examine=1,
        **kwargs,
    ):
        """
        Args:
          tokenizer:   A tokenizer for decoding tokens -> text
          hosts:       A list of host strings
          port:        Port to use for each sandbox
          num_examine: For printing / debugging
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.last_epoch = None
        self.last_batch_index = None

        port = 6000
        nodelist_str = os.environ.get("SLURM_JOB_NODELIST", "")
        hosts = expand_slurm_nodelist(nodelist_str)

        # Create one sandbox per host
        self.sandboxes = [get_sandbox(host=h, port=port, sandbox_type='fast') for h in hosts]

    def write_generations(self, dir_path, event, data: list[dict]):
        os.makedirs(dir_path, exist_ok=True)
        idx = str(uuid.uuid4())
        filepath = os.path.join(dir_path, f"{event}_{self.last_epoch}_{self.last_batch_index}_{idx}.jsonl")
        with open(filepath, 'w') as f:
            for item in data:
                print(json.dumps(item), file=f)

    def __call__(
        self,
        data: DataProto,
        epoch: int | None = None,
        batch_index: int | None = None,
        event: str | None = None,
        config: dict | None = None,
        ref_policy=None,
        return_reward_only: bool = True,
        logprob_reward_coef: float = 0.0,
        concurrency_per_sandbox: int = 16,  # <--- new
        item_results: list[dict] | None = None,
    ):
        """
        Main entry point.  We:
          1) Possibly short-circuit if 'rm_scores' in data
          2) Compute item_results (compute_item_results)
          3) Aggregate them prompt-by-prompt
          4) Possibly apply logprob shaping
          5) Return a reward_tensor or full suite
        """
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if item_results is None:
            concurrency_per_sandbox = config.reward_model.reward_fn_args.get('concurrency_per_sandbox', 16)
            item_results = self.compute_item_results(
                data=data, 
                config=config,
                event=event,
                concurrency_per_sandbox=concurrency_per_sandbox
            )


        # 0) track epoch/batch
        self.last_epoch = epoch
        self.last_batch_index = batch_index

        # 1) group items by prompt uid
        batches_indices: OrderedDict[str, list[int]] = OrderedDict()
        for i, item in enumerate(data):
            uid = item.non_tensor_batch['uid']
            batches_indices.setdefault(uid, []).append(i)

        # 3) Prepare reward, group_reward
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        group_reward_tensor = torch.zeros(reward_tensor.shape[0], device=reward_tensor.device, dtype=torch.float32)
        pass_ret = torch.zeros_like(reward_tensor)
        pass_tensor = torch.zeros(reward_tensor.shape[0], device=reward_tensor.device, dtype=torch.float32)
        prompt_pad_tokens = [None] * len(data)
        response_pad_tokens = [None] * len(data)
        non_reasoning_tokens = [None] * len(data)

        all_response_lengths = {}
        all_write_content = []
        metrics = [None] * len(data)

        # 4) aggregate item_results per group
        params_list = [
            (
                uid,
                idxes,
                data,
                item_results,
                event,
                config,
                self.tokenizer,
            )
            for uid, idxes in batches_indices.items()
        ]
        results = [_process_group(p) for p in params_list]

        for (
            uid,
            group_idxes,
            rewards,
            group_rewards,
            pass_or_not,
            num_non_reasoning_tokens_list,
            response_pad_local,
            prompt_pad_local,
            response_lengths,
            write_content,
            metrics_local,
        ) in results:

            all_write_content.append(write_content)
            all_response_lengths[uid] = response_lengths

            # Insert token-level rewards
            for i, idx in enumerate(group_idxes):
                item = data[idx]
                prompt_len = item.batch['prompts'].shape[-1]
                resp_len = item.batch['attention_mask'][prompt_len:].sum()

                reward_tensor[idx, resp_len - 1] = rewards[i]
                group_reward_tensor[idx]         = group_rewards[i]
                pass_ret[idx, resp_len - 1]      = int(pass_or_not[i])
                pass_tensor[idx]                 = int(pass_or_not[i])

                non_reasoning_tokens[idx] = num_non_reasoning_tokens_list[i]
                response_pad_tokens[idx]  = response_pad_local[i]
                prompt_pad_tokens[idx]    = prompt_pad_local[i]

                metrics[idx] = metrics_local[i]

        assert all((metrics[i] is not None for i in range(len(metrics)))), f'{metrics}'

        # 5) optional write
        if event in ('rollout', 'val'):
            out_dir = config.actor_rollout_ref.rollout.experience_output_dir
            self.write_generations(dir_path=out_dir, event=event, data=all_write_content)

        # turn them into Tensors
        non_reasoning_tokens = torch.tensor(non_reasoning_tokens, device=reward_tensor.device, dtype=torch.long)
        response_pad_tokens  = torch.tensor(response_pad_tokens, device=reward_tensor.device, dtype=torch.long)
        prompt_pad_tokens    = torch.tensor(prompt_pad_tokens, device=reward_tensor.device, dtype=torch.long)

        # summarize metrics
        # metrics = {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}

        # 6) optional logprob shaping
        if logprob_reward_coef > 0.0:
            logprob_rewards = self.generate_logprob_reward(
                config,
                data,
                non_reasoning_tokens,
                response_pad_tokens,
                ref_policy,
            )
            num_modified = 0
            for uid, idxes in batches_indices.items():
                # example: if we want to do some "ranking" or "mix"
                eligible = [(i, idx) for i, idx in enumerate(idxes) if logprob_rewards[idx] is not None]
                if len(eligible) <= 1:
                    continue
                vals = torch.tensor([logprob_rewards[idx] for _, idx in eligible], device=reward_tensor.device)
                vals -= vals.min()
                if vals.max() > 0:
                    vals /= vals.max()
                resp_lens = all_response_lengths[uid]
                for (grp_i, global_i), lv in zip(eligible, vals):
                    resp_idx = resp_lens[grp_i] - 1
                    if abs(float(reward_tensor[global_i, resp_idx])) < 1e-2:
                        reward_tensor[global_i, resp_idx] = logprob_reward_coef * lv
                        num_modified += 1
            print('log‑prob shaping applied to', num_modified, 'responses')

        # 7) special case 'val'
        if event == 'val':
            return pass_ret

        # 8) return
        if return_reward_only:
            return reward_tensor

        assert len(metrics) == reward_tensor.shape[0], f'{len(metrics)} vs {reward_tensor.shape[0]}'

        return (
            reward_tensor,
            group_reward_tensor,
            pass_tensor,
            non_reasoning_tokens,
            response_pad_tokens,
            prompt_pad_tokens,
            metrics,
        )

    def generate_logprob_reward(self, config, data, all_num_non_reasoning_tokens, response_num_pad_tokens, ref_policy):
        # same logic as before, or adapt as needed
        assert len(data) == len(all_num_non_reasoning_tokens)
        logprob_rewards = [None] * len(data)
        idx_solutions = []
        logprob_batch = {'input_ids': [], 'attention_mask': [], 'position_ids': [], 'responses': []}
        new_solution_len = []

        max_response_length = data.batch['responses'].shape[1]
        for i_batch in range(len(data)):
            input_ids      = data[i_batch].batch['input_ids']
            attention_mask = data[i_batch].batch['attention_mask']
            position_ids   = data[i_batch].batch['position_ids']
            response_ids   = data[i_batch].batch['responses']

            num_non_reasoning_tokens = all_num_non_reasoning_tokens[i_batch]
            prompt_len = input_ids[:-max_response_length].shape[-1]

            if response_num_pad_tokens[i_batch] > 0:
                amt = response_num_pad_tokens[i_batch]
                input_ids      = input_ids[:-amt]
                attention_mask = attention_mask[:-amt]
                position_ids   = position_ids[:-amt]
                response_ids   = response_ids[:-amt]

            if num_non_reasoning_tokens == response_ids.shape[-1] or num_non_reasoning_tokens == 0:
                continue

            reasoning_ids = response_ids[:-num_non_reasoning_tokens]
            prompt_and_reasoning_attention_mask = attention_mask[:-num_non_reasoning_tokens]
            prompt_and_reasoning_position_ids   = position_ids[:-num_non_reasoning_tokens]

            solutions = data[i_batch].non_tensor_batch['solutions']
            idx_solution = None
            for i_sol, lang_code in enumerate(solutions['language']):
                if lang_code == 3:  # e.g. python3
                    idx_solution = i_sol
                    break
            if idx_solution is None:
                continue

            solution_str = f"```python\n{solutions['solutions'][idx_solution]}\n```"
            sol_data = self.tokenizer(solution_str, return_tensors='pt', add_special_tokens=False)
            sol_token_ids = sol_data['input_ids'].to(device=input_ids.device)[0]
            sol_attention_mask = sol_data['attention_mask'].to(device=input_ids.device)[0]

            remaining_length = max_response_length - (reasoning_ids.shape[-1] + sol_token_ids.shape[-1])
            if remaining_length < 0:
                sol_token_ids = sol_token_ids[:remaining_length]
                sol_attention_mask = sol_attention_mask[:remaining_length]
                remaining_length = 0

            # pad up
            sol_token_ids = torch.cat([
                sol_token_ids,
                torch.ones(remaining_length, device=sol_token_ids.device, dtype=torch.long) * self.tokenizer.pad_token_id
            ], dim=-1)
            sol_attention_mask = torch.cat([
                sol_attention_mask,
                torch.zeros(remaining_length, device=sol_attention_mask.device, dtype=torch.long)
            ], dim=-1)

            delta_position_ids = torch.arange(1, sol_token_ids.shape[-1] + 1, device=sol_token_ids.device)
            idx_solutions.append(i_batch)
            new_solution_len.append(sol_token_ids.shape[-1])

            full_solution_ids = torch.cat([reasoning_ids, sol_token_ids], dim=-1)
            new_input_ids     = torch.cat([input_ids[:prompt_len], full_solution_ids], dim=-1)
            new_attention_mask= torch.cat([prompt_and_reasoning_attention_mask, sol_attention_mask], dim=-1)

            response_position_ids = prompt_and_reasoning_position_ids[-1] + delta_position_ids
            new_position_ids      = torch.cat([prompt_and_reasoning_position_ids, response_position_ids], dim=-1)

            logprob_batch['input_ids'].append(new_input_ids)
            logprob_batch['attention_mask'].append(new_attention_mask)
            logprob_batch['position_ids'].append(new_position_ids)
            logprob_batch['responses'].append(full_solution_ids)

        if len(idx_solutions) == 0:
            return logprob_rewards

        for k in logprob_batch:
            logprob_batch[k] = torch.stack(logprob_batch[k])
        num_samples = logprob_batch['input_ids'].shape[0]

        world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
        remainder = num_samples % world_size
        if remainder != 0:
            # replicate sample so total is multiple of world_size
            num_repeats = world_size - remainder
            for k in logprob_batch:
                repeated_sample = torch.stack([logprob_batch[k][0]] * num_repeats, dim=0)
                logprob_batch[k] = torch.cat([logprob_batch[k], repeated_sample], dim=0)

        # convert to DataProto
        logprob_batch = DataProto.from_single_dict(logprob_batch)
        log_probs = ref_policy.compute_ref_log_prob(logprob_batch)
        log_probs = log_probs.batch['ref_log_prob'][:num_samples]

        processed_log_probs = []
        for i in range(num_samples):
            sol_len = new_solution_len[i]
            chunk = log_probs[i, -sol_len:]
            processed_log_probs.append(chunk.mean().item())

        for i, j in enumerate(idx_solutions):
            logprob_rewards[j] = processed_log_probs[i]

        return logprob_rewards
