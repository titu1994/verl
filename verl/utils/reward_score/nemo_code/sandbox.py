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

import abc
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

LOG = logging.getLogger(__file__)

import cProfile
import io
import pstats


class Sandbox(abc.ABC):
    """Code execution sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        host: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1"),
        port: Optional[str] = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000"),
        ssh_server: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.http_session = requests.Session()
        self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH", ssh_key_path)
        # will keep state of code sessions
        self.sessions = {}

    def clear_session(self, session_id):
        del self.sessions[session_id]

    def _send_request(self, request, timeout):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            output = sshtunnel_request.post(
                url=self._get_execute_url(),
                data=json.dumps(request),
                timeout=(timeout, timeout),
                headers={"Content-Type": "application/json"},
            )
        else:
            output = self.http_session.post(
                url=self._get_execute_url(),
                data=json.dumps(request),
                timeout=(timeout, timeout),
                headers={"Content-Type": "application/json"},
            )

        # retrying 502 errors
        if output.status_code == 502:
            raise requests.exceptions.Timeout
        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output):
        pass

    @abc.abstractmethod
    def _get_execute_url(self):
        pass

    @abc.abstractmethod
    def _prepare_request(self, generated_code, timeout):
        pass

    def profile_execute_code(self, *args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        # Run the function you want to profile
        result = self.execute_code(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()

        # Print profiling results
        print(s.getvalue())

        return result

    def execute_code(
        self,
        generated_code: str,
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict, str]:
        if session_id is None:  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []
        generated_code = generated_code.replace('"""', r'\"\"\"')
        self.sessions[session_id].append(generated_code)
        TO_EXECUTE = """
import traceback
import json
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '16'

code_snippets = []
"""
        for code_snippet in self.sessions[session_id]:
            TO_EXECUTE += f'\ncode_snippets.append("""{code_snippet}""")\n'

        TO_EXECUTE += f"""
try:
    for code in code_snippets:
        stdout = stderr = ""
        exec_locals, exec_globals = {{}}, {{}}
        exec(code, exec_globals, exec_locals)
        stdout += "\\n".join([str(v) for v in exec_locals.values()])
    if len(stdout) > {max_output_characters}:
        stdout = stdout[:{max_output_characters}] + "<output cut>"
    if len(stderr) > {max_output_characters}:
        stderr = stderr[:{max_output_characters}] + "<output cut>"
    to_return = {{"process_status": "completed", "stdout": stdout, "stderr": stderr}}
except Exception:
    # removing useless prefix from traceback
    to_return = {{
        "process_status": "error",
        "stdout": "",
        "stderr": traceback.format_exc(),
    }}
print(json.dumps(to_return))
"""
        request = self._prepare_request(TO_EXECUTE, timeout)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
        # removing last state to not re-execute code with errors
        if output['stderr'] != "":
            self.sessions[session_id] = self.sessions[session_id][:-1]
        return output, session_id

    def is_output_correct(self, pred_output, gt_output, include_percentage=True, tolerance=1e-4, timeout=10.0):
        # embedding the full math grader code here to send to server for execution
        with open(Path(__file__).absolute().parent / "math_grader.py", "rt") as fin:
            math_grader_code = fin.read()

        # corner cases
        if isinstance(pred_output, str):
            pred_output = pred_output.replace("'''", r'\'\'\'')
            while pred_output.endswith('\\'):
                pred_output = pred_output[:-1]

        if isinstance(gt_output, str):
            gt_output = gt_output.replace("'''", r'\'\'\'')
            while gt_output.endswith('\\'):
                gt_output = gt_output[:-1]

        TO_EXECUTE = f"""
import os
import sys
import json
from io import StringIO
os.environ['OPENBLAS_NUM_THREADS'] = '16'

{math_grader_code}

stdout = sys.stdout
# removing all output to not capture that
sys.stdout = sys.stderr = StringIO()
try:
    output = math_equal(
        r'''{pred_output}''',
        r'''{gt_output}''',
        {include_percentage},
        {tolerance},
        {timeout},
    )
    error_message = ""
except Exception as e:
    output = False
    error_message = str(e)
# restoring the output to get the print
sys.stdout = stdout
print(json.dumps({{"result": output, "error_message": error_message}}))
"""
        request = self._prepare_request(TO_EXECUTE, timeout)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {'result': False, 'error_message': Sandbox.TIMEOUT_ERROR}
        if output['error_message']:
            # logging the error
            LOG.warning("Error during correctness check: %s", output['error_message'])

        return output['result']


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self):
        return f"http://{self.host}:{self.port}/execute"

    def _parse_request_output(self, output):
        return output.json()

    def _prepare_request(self, generated_code, std_input: str = "", timeout: float = 3, language: str = 'python'):
        return {
            "generated_code": generated_code,
            "timeout": timeout,
            "std_input": std_input,
            "language": language,
        }


class FastLocalSandbox(LocalSandbox):
    """Locally hosted sandbox."""

    def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: Optional[str] = None,
        language: str = 'python',
    ) -> Tuple[Dict, str]:
        if session_id is None:  # creating a new session with empty state
            session_id = uuid.uuid4()
            self.sessions[session_id] = []
        self.sessions[session_id].append(generated_code)

        request = self._prepare_request(generated_code, std_input, timeout, language)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {
                "process_status": "timeout",
                "stdout": "TimeoutError",
                "stderr": "TimeoutError",
                "traceback": "TimeoutError",
            }
        # removing last state to not re-execute code with errors
        if output['stderr'] != "":
            self.sessions[session_id] = self.sessions[session_id][:-1]
        return output, session_id

    def run_checker(
            self,
            cf_contest_id: int,
            cf_index: str,
            inputs: str,
            actual_outputs: str,
            expected_outputs: str,
            timeout: float = 10.0
    ) -> Dict[str, str]:
        url = f"http://{self.host}:{self.port}/checker"

        request = {
            'cf_contest_id': cf_contest_id,
            'cf_index': cf_index,
            'inputs': inputs,
            'expected_outputs': expected_outputs,
            'actual_outputs': actual_outputs,
            'timeout': timeout,
        }

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            output = sshtunnel_request.post(
                url=url,
                data=json.dumps(request),
                timeout=(timeout, timeout),
                headers={"Content-Type": "application/json"},
            )
        else:
            output = self.http_session.post(
                url=url,
                data=json.dumps(request),
                timeout=(timeout, timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._parse_request_output(output)


class PistonSandbox(Sandbox):
    """Piston sandbox (https://github.com/engineer-man/piston)"""

    def _get_execute_url(self):
        return f"{self.host}/execute"

    def _parse_request_output(self, output):
        output = output.json()
        if output['run']['signal'] == "SIGKILL":
            return {'result': None, 'error_message': 'Unknown error: SIGKILL'}
        return json.loads(output['run']['output'])

    def _prepare_request(self, generated_code, timeout):
        return {
            "language": "py",
            "version": "3.10.0",
            "files": [
                {
                    "content": generated_code,
                }
            ],
            "stdin": "",
            "args": [],
            "run_timeout": timeout * 1000.0,  # milliseconds
            "compile_memory_limit": -1,
            "run_memory_limit": -1,
        }


sandboxes = {
    'local': LocalSandbox,
    'fast': FastLocalSandbox,
    'piston': PistonSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)
