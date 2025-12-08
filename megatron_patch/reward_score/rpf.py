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
from megatron.training import get_args, pretrain, print_rank_0
import re

def extract_answer_content(text):
    pattern = r'</think>(.*?)$'
#     print(text,"text")
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return "未找到答案内容"

# from mathruler.grader import extract_boxed_content, grade_answer
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

def analyze_mixed(text :str):
    chinese_count = 0
    english_count = 0
    
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            chinese_count += 1
        elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
            english_count += 1
    
    total = chinese_count + english_count
    if total == 0:
        return None
    
    mixed_ratio = (chinese_count * english_count) / total  # 交叉占比，值越大混杂程度越高
    return {
        "is_mixed": chinese_count > 0 and english_count > 0,
        "chinese_ratio": chinese_count / total,
        "english_ratio": english_count / total,
        "mixed_ratio": mixed_ratio
    }


def clean_content(text : str):
    if '<think>' in text:
        text = text.replace("<think>",'')
    if '</think>' in text:
        text = text.replace('</think>','')
    return text


def language_reward(predict_str: str, ground_truth: str):
    predict_str = clean_content(predict_str)
    groud_truth = clean_content(ground_truth)

    predict_dict = analyze_mixed(predict_str)
    ground_dict = analyze_mixed(groud_truth)

    language_score = 0.0
    if predict_dict and ground_dict:
        # mixed
        if predict_dict['is_mixed'] == ground_dict['is_mixed']:
            if predict_dict['is_mixed'] == True:
                # mixed language
                language_score = -1.0
            else:
                # same language
                languge_score = 1.0
    return language_score


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r'<think>.*</think>.*\\boxed\{.*\}.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else -1.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(extract_answer_content(predict_str))
    reward = 0.0
    if answer.strip() == ground_truth.strip():
        reward += 0.5
    else:
        reward -= 0.5
    gt = "\\boxed{"+ground_truth+"}"
    if predict_str.find(gt)!=-1:
        reward += 0.5
    else:
        reward -= 0.5
    return reward 


def compute_score(solution_str, ground_truth) -> float:
    # print_rank_0(extract_answer_content(solution_str))
    extract_answer_content(solution_str)
    answer = extract_boxed_content(extract_answer_content(solution_str))
    score1 = acc_reward(solution_str, ground_truth) * 0.9
    score2 = 0.1 * format_reward(solution_str)
    score = score1 + score2
    # print_rank_0(answer, ground_truth, score, score1, score2)
    style = 'rule'
    return score
    # return {
    #     "score": score,
    #     "style" : style
    #     }




# a= '''<think>
# Okay, let's tackle this problem. The user provided a text with a [MASK] token that needs to be predicted. The context is a patent office review opinion regarding the creativity of a tire blackening agent. The task is to figure out what the masked token is, list multiple possibilities, and select the most probable one.

# First, I need to understand the context around the [MASK]. The text is discussing the technical features of the patent application and comparing it with a prior art document (Contrast File 1). The specific part with the [MASK] is in the section discussing the differences between the claimed invention and the prior art. The sentence says: "其[MASK]技术障碍，难以从说明书中看出其表面活性剂的特定选择可以给本申请带来何种预料不到的技术效果..." which translates to something like "its [MASK] technical obstacle, it is difficult to see from the description how the specific selection of surfactants can bring unexpected technical effects..."

# Looking at the surrounding text, the paragraph is explaining that the differences between the claimed invention and the prior art are the specific surfactants (two-ion surfactant, alkylbenzenesulfonate sodium, pH regulator, chelating agent) and the amounts of each component. The argument is that these differences are considered obvious to a person skilled in the art because the components are common in the field, so there's no technical obstacle that would make their specific selection unexpected.

# The sentence with [MASK] is stating that there's a technical obstacle, but the text suggests that the selection of surfactants is not novel. Wait, but the text says that the specific selection of surfactants is not obvious. Wait, no. The text says that the specific selection of the surfactants (two-ion surfactants, alkylbenzenesulfonate sodium, etc.) is considered common in the field, so there's no technical obstacle. However, the sentence here is saying that there's a technical obstacle, but the text is trying to argue that there isn't one. Wait, maybe I need to look at the exact wording again.

# The original text after the [MASK] says: "其[MASK]技术障碍，难以从说明书中看出其表面活性剂的特定选择可以给本申请带来何种预料不到的技术效果..." So the sentence is saying that there's a [MASK] technical obstacle, making it difficult to see from the description how the specific surfactant selection leads to unexpected effects. But the rest of the text argues that there is no such technical obstacle because the surfactants are common.

# Wait, that's conflicting. The paragraph is trying to say that the specific surfactant selection is not novel, so there's no technical barrier. But the sentence with [MASK] is stating that there is a technical obstacle. Maybe the [MASK] is a word that fits in the context of "technical barrier" but that's not making sense. Wait, maybe the [MASK] is part of a phrase like "存在技术障碍" (there exists a technical obstacle), but the text is arguing that there isn't. Hmm.

# Alternatively, maybe the [MASK] is a word that's missing in the sentence. The sentence structure is: "其[MASK]技术障碍，难以从说明书中看出其表面活性剂的特定选择可以给本申请带来何种预料不到的技术效果..." So the part before "技术障碍" is missing. The sentence is trying to say that there's a technical obstacle that makes it impossible to see from the description how the specific surfactant selection brings unexpected effects. However, the rest of the text argues that such a technical obstacle doesn't exist because those components are common.

# Wait, this is confusing. The paragraph is actually arguing that the specific selection of surfactants is not novel, so there's no technical obstacle. Therefore, the [MASK] here might be a word that is part of a phrase that contradicts the rest of the text. But that doesn't make sense. Alternatively, perhaps the [MASK] is part of a phrase that should be completed as "不存在" (does not exist), but the sentence is saying "there is a [MASK] technical obstacle", which would mean that the [MASK] is "存在" (exists). But that would contradict the rest of the text, which is arguing that there is no such obstacle.

# Wait, maybe the text is structured as follows: The paragraph is saying that the specific selection of surfactants is not novel, so there's no technical obstacle. However, the sentence with [MASK] is structured as "其[MASK]技术障碍，难以从说明书中看出..." which is saying that there is a technical obstacle that makes it hard to see... But that contradicts the rest of the paragraph. Therefore, perhaps the author made a mistake, but as the task is to predict the token, we need to look for the most probable word that fits the context.

# Alternatively, maybe the sentence is trying to say that the specific selection of surfactants has a technical obstacle, but the rest of the text is explaining why that's not true. Maybe the [MASK] is "存在" (exists), but that's not correct. Alternatively, maybe the [MASK] is "遗漏" (omitted) or "未被提及" (unmentioned), but that's not the case.

# Alternatively, the sentence might be missing a word that is part of a common phrase. For example, in Chinese, a common phrase could be "存在技术障碍" (there exists a technical obstacle), but the text is trying to argue that there is no such obstacle. However, the structure of the sentence is "其[MASK]技术障碍，难以从说明书中看出..." So the [MASK] is a word that would make sense with "技术障碍". Let me think of possible words that fit. The possible words could be: 存在 (exist), 无 (none), 有 (there is), 等 (such as). Given the context, the text is arguing that there is no technical obstacle, so the [MASK] should be "无" (none). But the sentence is structured as "其[MASK]技术障碍，难以从说明书中看出..." If the [MASK] is "无", then the full sentence is "其无技术障碍，难以从说明书中看出..." which would mean "There is no technical obstacle, it is difficult to see from the description..." But this would be contradictory because the rest of the paragraph is saying that there is no technical obstacle. However, the sentence is part of a larger argument where the author is trying to say that the specific selection of surfactants is not novel, thus there is no technical obstacle. But the sentence as written is "其[MASK]技术障碍，难以从说明书中看出..." which would be "There is [masked] technical obstacle, it is difficult to see from the description..." This seems to be part of an argument that the specific selection has no technical obstacle. Therefore, the [MASK] should be "存在" (there is), but the text is arguing that there isn't. Wait, this is confusing.

# Alternatively, perhaps the [MASK] is a word that is part of the sentence's structure. For example, maybe the correct word is "存在", but the text is trying to say that there is no technical obstacle. Wait, but the author is trying to make the point that the selection is obvious, hence no technical obstacle. Therefore, the sentence should be "其不存在技术障碍..." (there is no technical obstacle). But in the text, the sentence says "其[MASK]技术障碍，难以从说明书中看出..." So the [MASK] would be "不存在". That would make the sentence: "其不存在技术障碍，难以从说明书中看出..." which means "There is no technical obstacle, it is difficult to see from the description..." But that doesn't make sense. Wait, perhaps the sentence is saying that the existence of the technical obstacle makes it difficult to see... but that would be contradictory. Alternatively, maybe the correct word is "存在", and the author is making an argument that even though there is a technical obstacle, the description doesn't explain it. But that doesn't fit the rest of the paragraph.

# Looking at the context again, the paragraph is arguing that the specific surfactant selection is not novel because these are common components in the field, so there's no technical obstacle. Therefore, the sentence should be stating that there is no technical obstacle, hence the [MASK] is "不存在". But the wording is "其[MASK]技术障碍，难以从说明书中看出..." which would be "There is [masked] technical obstacle, it's hard to see from the description..." If the [MASK] is "不存在", then the sentence would be "There is no technical obstacle, it's hard to see from the description..." which doesn't make sense. Alternatively, maybe the [MASK] is "存在", but that would contradict the argument. This is really tricky.

# Alternatively, maybe the [MASK] is a word that is part of a common phrase. For example, in Chinese legal text, terms like "存在技术障碍" (there exists a technical obstacle) are common. However, in the context, the argument is that there's no such obstacle. Therefore, the correct [MASK] would be "不存在", but the sentence structure would be incorrect. Alternatively, maybe the [MASK] is "有", making the sentence "其有技术障碍..." which would mean "There is a technical obstacle..." but the text is arguing that there isn't one. This is very confusing.

# Given the options, perhaps the most probable answer is "存在", but that contradicts the rest of the paragraph. Alternatively, the correct term might be "不存在", which would make the sentence "其不存在技术障碍..." but the structure of the sentence is incorrect. Alternatively, maybe the [MASK] is part of a phrase like "具有技术障碍", but that's not matching the structure.

# Alternatively, the sentence might be saying that the specific selection's technical obstacle is not present, hence the [MASK] is "无", making the sentence "其无技术障碍，难以从说明书中看出..." which would mean "There is no technical obstacle, it is difficult to see from the description..." but that still doesn't make sense. Alternatively, maybe the structure is wrong, and the correct [MASK] is "存在", but the text's argument is that there isn't one.

# In this case, I need to make the best possible prediction. Given the context, the paragraph is arguing that the specific selection is not novel, so there's no technical obstacle. Therefore, the [MASK] should be "不存在" (does not exist). However, the sentence structure would be "其不存在技术障碍，难以从说明书中看出..." which would mean "There is no technical obstacle, it's hard to see from the description..." but this doesn't align logically. Alternatively, maybe the [MASK] is "存在", but that would be contradictory. However, considering that the sentence is part of a larger argument where the author is trying to say that the specific selection is not new, the correct term would be "不存在", even though the sentence structure might be awkward.

# Alternatively, perhaps the correct answer is "存在", and the rest of the paragraph is trying to argue that despite this, the specific selection is not novel. But that's not matching. This is really challenging.

# Another approach: Look for similar phrases in legal texts. In Chinese patent reviews, a common phrase would be "存在技术障碍" (there is a technical obstacle). But in the context here, the argument is that there isn't one. Therefore, the [MASK] is "不存在". Even though the sentence structure is a bit off, this might be the correct answer.

# Alternatively, the [MASK] could be "有", making the sentence "其有技术障碍..." but that would mean "there is a technical obstacle..." which contradicts the argument.

# Given all this, the most probable answer, even if the sentence structure is a bit awkward, is that the [MASK] is "存在", and the text is trying to argue that despite the existence of a technical obstacle, the specific selection is not novel. But this is not coherent. Alternatively, the [MASK] is "不存在", even though the sentence structure is incorrect.

# But the user wants the most probable token. In the context, the author is explaining that the specific selection is not novel, so there's no technical obstacle. Therefore, the [MASK] is "不存在". Hence, the predicted token is "不存在".
# </think>

# The masked token is part of a sentence discussing the technical feasibility of the patent application. The context indicates that the specific selection of surfactants (e.g., two-ion surfactants, alkylbenzenesulfonate sodium) is considered common in the field, making the technical obstacle non-existent. The sentence structure implies that the lack of a technical obstacle makes it difficult to observe unexpected effects from the surfactant selection. 

# **Potential tokens:**  
# - 存在 (exist)  
# - 无 (none)  
# - 不存在 (does not exist)  
# - 有 (there is)  

# **Most probable token:**  
# The phrase "其[MASK]技术障碍" is part of an argument that the specific surfactant selection is not novel. The rest of the text explicitly states that these components are common, implying **no technical obstacle**. While the sentence structure is slightly awkward, the logical conclusion is that the technical obstacle **does not exist**. 

# \\boxed{ 不存在}'''
# b= "不存在"
# print(compute_score(a, b))