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

import re
# from mathruler.grader import extract_boxed_content, grade_answer
from megatron_patch.reward_score.instruct_eval import parse_and_eval

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r'<think>.*</think>.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0

def extract_answer_content(text):
    pattern = r'</think>(.*?)$'
#     print(text,"text")
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return "未找到答案内容"

def acc_reward(predict_str: str, ground_truth: str) -> float:

    # answer = extract_answer_content(predict_str)
    # print("xxx"+ground_truth[0]+"xxx")
    # print("xxx"+type(ground_truth[0])+"xxx")
    try:
        score = parse_and_eval(ground_truth,predict_str)
    except:
        score = 0.0
#     print("====")
#     print(score)
#     print("====")
    # print(score)
    return score
#     return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(solution_str: str, ground_truth: str) -> float:
    acc = 0.9 * acc_reward(solution_str, ground_truth) + 0.1 * format_reward(solution_str)
    reward = 1.0*acc if acc>0.5 else -1.0*acc
    

    return reward
