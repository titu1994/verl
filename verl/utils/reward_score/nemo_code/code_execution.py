# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import io
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback

from verl.utils.reward_score.nemo_code.sandbox import get_sandbox


def execute(input_dict):
    output_dicts = []
    for _, entry in enumerate(input_dict['prompts']):
        get_output = re.findall(r'```(.*?)```', entry['output'], re.DOTALL)
        completion = get_output[0] if len(get_output) > 0 else entry['output']
        completion = re.sub(r'^(?:\n)?[Pp]ython\s*', '', completion)
        completion += '\n'

        if input_dict['local_execution']:
            return_dict = local_code_execution(completion, entry['unit_tests'], input_dict['code_execution_timeout'])
        else:
            return_dict = sandbox_code_execution(completion, entry['unit_tests'], input_dict['code_execution_timeout'])
        output_dicts.append(return_dict)

    return output_dicts


def sandbox_code_execution(completion, unit_tests, timeout=2, sandbox=None):

    output_dict = {
        "correct_tests": [],
        "average_test_score": 0.0,
        "unit_test_stdouts": [],
        "unit_test_stderrs": [],
        "traceback": [],
        "time_taken": [],
    }
    start = time.time()
    if sandbox is None:
        sandbox = get_sandbox('fast', host=os.getenv('NEMO_SKILLS_SANDBOX_HOST', 'localhost'))

    for _, inp in enumerate(unit_tests):
        try:
            if isinstance(inp, str):
                # The unit test is a string assert test case
                output, _ = sandbox.execute_code(completion + inp, timeout=timeout)

            elif isinstance(inp, dict):
                # The unit test is a dictionary of input and expected output
                if not "input" in inp or "output" not in inp:
                    raise Exception("Test dictionary should have 'input' and 'output' keys")

                # The unit test is a dictionary of input and expected output
                output, _ = sandbox.execute_code(completion, std_input=inp['input'], timeout=timeout)

                # Do exact match but with lenience for trailing whitespaces
                correct_tests = inp['output'].strip().strip("\n") == output['stdout'].strip().strip("\n")

                # Check if the output is correct
                if correct_tests:
                    output['stderr'] = ''
                else:
                    # Explicitly set stderr to 'AssertionError' if the output is incorrect
                    output['stderr'] = output['stderr'] or 'AssertionError'

            else:
                raise Exception("Unit test should be either a string or a dictionary (with keys 'input' and 'output')")

            if output['process_status'] == 'timeout':
                print("Timeout error for entry: ", completion + inp, output)

            output_dict['correct_tests'].append(output['stderr'] == '')
            output_dict['unit_test_stderrs'].append(output['stderr'])
            output_dict['traceback'].append(output['traceback'])
            output_dict['unit_test_stdouts'].append(output['stdout'])

        except Exception as e:
            output_dict['correct_tests'].append(False)
            output_dict['unit_test_stderrs'].append(repr(e))
            output_dict['traceback'].append(repr(e))
            output_dict['unit_test_stdouts'].append("")
        finally:
            output_dict['time_taken'].append(time.time() - start)

    print("Time taken for all unit tests in entry: ", time.time() - start)
    # Calculate the average test score
    output_dict['average_test_score'] = (
        0.0
        if len(output_dict['correct_tests']) == 0
        else (sum(output_dict['correct_tests']) / len(output_dict['correct_tests']))
    )

    return output_dict


def local_code_execution(completion, unit_tests, timeout=3):
    output_dict = {
        "correct_tests": [],
        "average_test_score": 0.0,
        "unit_test_stdouts": [],
        "unit_test_stderrs": [],
        "traceback": [],
        "time_taken": [],
    }

    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        custom_globals = {"__builtins__": __builtins__}

        for _, inp in enumerate(unit_tests):
            start = time.time()

            custom_globals = {"__builtins__": __builtins__}
            try:
                with time_limit(timeout):
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    exec(completion + inp, custom_globals)

                err = sys.stderr.getvalue()
                output_dict['correct_tests'].append(err == '')
                output_dict['unit_test_stderrs'].append(err)
                output_dict['traceback'].append("\n".join(traceback.format_exc().split("\n")[3:]))

            except Exception as e:
                output_dict['correct_tests'].append(False)
                output_dict['unit_test_stderrs'].append(repr(e))
                output_dict['traceback'].append("\n".join(traceback.format_exc().split("\n")[3:]))

            finally:
                out = sys.stdout.getvalue()
                err = sys.stderr.getvalue()
                output_dict['time_taken'].append(time.time() - start)
                output_dict['unit_test_stdouts'].append(out)

        output_dict['average_test_score'] = (
            0.0
            if len(output_dict['correct_tests']) == 0
            else (sum(output_dict['correct_tests']) / len(output_dict['correct_tests']))
        )

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

        return output_dict


class TimeoutError(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname
