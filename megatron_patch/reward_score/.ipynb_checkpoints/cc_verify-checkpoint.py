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
# from megatron_patch.reward_score.instruct_eval import parse_and_eval
def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"
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
    # boxed = extract_boxed_content(answer)
    if "不具备新颖性" not in predict_str and "不具备创造性" not in predict_str:
        boxed ="A"
    else:
        boxed="X"
    if boxed.strip()==ground_truth.strip():
        score = 1.0
    else:
        score = -1.0
    return score

def extract_markdown_titles(text, level=3, target_titles=None):
    """
    灵活抽取Markdown标题
    
    Args:
        text: 文章文本
        level: 标题级别（1-6）
        target_titles: 目标标题列表，如果为None则抽取所有该级别的标题
    """
    # 构建正则表达式
    hash_marks = '#' * level
    
    if target_titles:
        # 如果指定了目标标题，只匹配这些标题
        targets = '|'.join(re.escape(title) for title in target_titles)
        pattern = f'^{hash_marks}\\s*({targets})\\s*$'
    else:
        # 匹配所有该级别的标题
        pattern = f'^{hash_marks}\\s+(.+?)\\s*$'
    
    titles = re.findall(pattern, text, re.MULTILINE)
    return titles
    
def title_reward(predict_str: str) -> float:

    answer = extract_answer_content(predict_str)
    boxed = extract_boxed_content(answer)
    titles = ['一、申请专利的技术特征分析', '二、对比文件的技术特征分析', '三、技术特征对比', '四、区别技术特征分析', '五、区别技术特征是否为本领域常规手段','六、结论']
    specific_titles = extract_markdown_titles(predict_str, level=3, target_titles=titles)

    score = float(len(list(set(specific_titles)))/len(list(set(titles))))
    # if score<0.5:
    #     score = -1.0*score
    
    return score

def compute_score(solution_str: str, ground_truth: str) -> float:
    r1 = acc_reward(solution_str, ground_truth)
    # r2 = format_reward(solution_str) 
    # r3 = title_reward(solution_str) 
    # acc = 0.9 * r1 + 0.1 * r2 
    acc = r1

    reward = acc
    

    return reward





