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

import multiprocessing
import os
import resource
import subprocess
import tempfile
import time
import traceback
import sys
from io import StringIO
from multiprocessing import TimeoutError

from flask import Flask, request

app = Flask(__name__)


class ExitExecution(Exception):
    pass


def execute_code(generated_code, std_input, timeout, language):
    """
    Execute code via writing files and calling external interpreters/compilers.
    Supports both C++ and Python.
    """
    def set_limits():
        """Set resource limits for the subprocess."""
        limit = 1024 * 1024 * 1024 * 10  # 10 GB
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    # Create a temporary file to write the stdin input
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_input:
        input_file_path = temp_input.name
        temp_input.write(std_input)
        temp_input.flush()  # Ensure all content is written to the file

    # Create a temporary file to store the generated code
    file_suffix = ".cpp" if language == "cpp" else ".py"
    with tempfile.NamedTemporaryFile(mode='w', suffix=file_suffix, delete=False) as temp_code:
        code_file_path = temp_code.name
        temp_code.write(generated_code)
        temp_code.flush()  # Ensure all content is written to the file

    binary_file_path = None
    if language == "cpp":
        binary_file_path = tempfile.NamedTemporaryFile(delete=False).name

    try:
        if language == "cpp":
            # Compile the C++ code using g++
            compile_process = subprocess.run(
                ["g++", code_file_path, "-o", binary_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )

            # Check for compilation errors
            if compile_process.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": compile_process.stderr,
                    "traceback": "Compilation failed."
                }

            # Set the command to execute the compiled binary
            execution_command = [binary_file_path]

        elif language == "python":
            # Set the command to execute the Python script
            execution_command = ["python3", code_file_path]

        else:
            return {
                "stdout": "",
                "stderr": "Unsupported language specified.",
                "traceback": "Invalid language."
            }

        # Run the code in a subprocess with resource limits
        process = subprocess.Popen(
            execution_command,
            stdin=open(input_file_path, "r"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            preexec_fn=set_limits  # Set resource limits in the subprocess
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "traceback": "",
        }

    finally:
        # Cleanup temporary files
        try:
            os.remove(input_file_path)
            os.remove(code_file_path)
            if binary_file_path:
                os.remove(binary_file_path)
        except OSError as e:
            print(f"Error cleaning up temporary files: {e}")


def execute_code_subprocess(generated_code, queue, std_input=None):
    """
    Execute Python code in-memory using exec() with resource limits.
    This is used if no std_input is provided to the /execute API.
    """
    # Set memory and CPU limits to avoid server crashes
    # Note: the limit below is 2 GB (adjust as needed)
    limit = 1024 * 1024 * 1024 * 2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    resource.setrlimit(resource.RLIMIT_STACK, (limit, limit))
    limit_cpu = 5
    resource.setrlimit(resource.RLIMIT_CPU, (limit_cpu, limit_cpu))

    stdout_2 = StringIO()
    sys.stdout = stdout_2  # Redirect stdout

    local_vars = {
        "stdout_2": stdout_2,
        "__name__": '__main__',
    }

    # Override exit and sys.exit so that the program ends gracefully
    def custom_exit(*args):
        sys.stdout.flush()
        stdout_2.flush()
        raise ExitExecution()

    local_vars["exit"] = custom_exit
    sys.exit = custom_exit

    if std_input:
        sys.stdin = StringIO(std_input)

    try:
        exec(generated_code, local_vars, None)
        queue.put({"stdout": stdout_2.getvalue(), "stderr": "", "traceback": ""})
    except ExitExecution:
        queue.put({"stdout": stdout_2.getvalue(), "stderr": "", "traceback": ""})
    except Exception as e:
        queue.put({
            "stdout": stdout_2.getvalue(),
            "stderr": f"{type(e).__name__}: {str(e)}",
            "traceback": "\n".join(traceback.format_exc().split("\n")[3:]),
        })


@app.route("/execute", methods=["POST"])
def execute():
    generated_code = request.json['generated_code']
    std_input = request.json.get('std_input', "")
    timeout = request.json['timeout']
    language = request.json['language']

    try:
        start = time.time()

        # If the language is Python and no standard input was provided,
        # run the code using our faster in-memory exec() logic.
        if language.lower() == "python" and not std_input:
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=execute_code_subprocess,
                args=(generated_code, queue, std_input)
            )
            process.start()
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
                if process.is_alive():
                    process.kill()
                return {
                    "process_status": "timeout",
                    "execution": "FAILED",
                    "stdout": "TimeoutError",
                    "stderr": "TimeoutError",
                    "traceback": "TimeoutError",
                }
            output = queue.get() if not queue.empty() else {"stdout": "", "stderr": "", "traceback": ""}
        else:
            # Otherwise (or if std_input is provided), use input/output write-based logic.
            output = execute_code(generated_code, std_input, timeout, language.lower())

        print(f"Time taken for execution {(time.time() - start):.2f}s")
        return {
            "process_status": "completed",
            "execution": "SUCCESS",
            "stdout": output.get("stdout", ""),
            "stderr": output.get("stderr", ""),
            "traceback": output.get("traceback", ""),
        }
    except TimeoutError:
        return {
            "process_status": "timeout",
            "execution": "FAILED",
            "stdout": "TimeoutError",
            "stderr": "TimeoutError",
            "traceback": "TimeoutError",
        }
    except Exception as e:
        return {
            "process_status": "error",
            "execution": "FAILED",
            "stdout": "",
            "stderr": str(e),
            "traceback": traceback.format_exc(),
        }


def checker(cf_contest_id, cf_index, inputs, actual_outputs, expected_outputs, queue):
    binary_path = f"/app/checkers/{cf_contest_id}/{cf_index}/check"

    try:
        # Create temporary files for inputs, actual_outputs, and expected_outputs
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_input, \
             tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_actual, \
             tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_expected:

            # Write string data to the temp files
            temp_input.write(inputs)
            temp_actual.write(actual_outputs)
            temp_expected.write(expected_outputs)

            # Get the file paths
            temp_input_path = temp_input.name
            temp_actual_path = temp_actual.name
            temp_expected_path = temp_expected.name

        # Execute the binary with the temporary file paths
        result = subprocess.run(
            [binary_path, temp_input_path, temp_actual_path, temp_expected_path],
            text=True,  # Capture output as text
            capture_output=True,  # Capture stdout and stderr
            check=True,  # Raise exception on non-zero exit
        )

        # Send success result back through the queue
        queue.put(
            {
                "stdout": "SUCCESS",
                "stderr": result.stderr,
                "traceback": None,
            }
        )
    except subprocess.CalledProcessError as e:
        # Send error result back through the queue
        queue.put(
            {
                "stdout": "FAILED",
                "stderr": e.stderr,
                "traceback": str(e),
            }
        )
    except Exception as e:
        # Handle other exceptions and send traceback
        queue.put(
            {
                "stdout": "ERROR",
                "stderr": "",
                "traceback": str(e),
            }
        )
    finally:
        # Clean up temporary files
        for temp_file in [temp_input_path, temp_actual_path, temp_expected_path]:
            try:
                os.remove(temp_file)
            except OSError:
                pass


@app.route("/checker", methods=["POST"])
def execute_checker():
    cf_contest_id = request.json['cf_contest_id']
    cf_index = request.json['cf_index']
    inputs = request.json['inputs']
    expected_outputs = request.json['expected_outputs']
    actual_outputs = request.json['actual_outputs']
    timeout = request.json['timeout']

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=checker,
        args=(cf_contest_id, cf_index, inputs, actual_outputs, expected_outputs, queue),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)
        if process.is_alive():
            process.kill()
        return {
            "process_status": "timeout",
            "stdout": "TimeoutError",
            "stderr": "TimeoutError",
            "traceback": "TimeoutError",
        }

    result = queue.get() if not queue.empty() else {}
    return {
        "process_status": "completed",
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "traceback": result.get("traceback", ""),
    }
