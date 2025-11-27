import re
from typing import List

def _split_children(content: str) -> List[str]:
    """
    一个辅助函数，用于将一个意图的内容字符串
    分割为其顶层的子块 (e.g., [SL:...] 或 [IN:...])
    
    它可以正确处理嵌套括号。
    """
    children = []
    bracket_level = 0
    current_start = -1

    for i, char in enumerate(content):
        if char == '[':
            if bracket_level == 0:
                current_start = i
            bracket_level += 1
        elif char == ']':
            bracket_level -= 1
            if bracket_level == 0 and current_start != -1:
                # 找到了一个完整的顶层子块
                children.append(content[current_start : i + 1])
                current_start = -1
    return children

def canonicalize_lf(lf: str) -> str:

    lf = lf.strip().lower()
    
    if not lf.startswith('['):
        return lf

    try:
        tag_end = lf.index(':')
        
        name_end = -1
        for i in range(tag_end + 1, len(lf)):
            if lf[i] == ' ':
                name_end = i
                break
            if lf[i] == ']': 
                name_end = i
                break
        
        if name_end == -1: return lf 

        tag = lf[1:tag_end]        
        name = lf[tag_end+1:name_end] 
        
        content = lf[name_end+1:-1].strip()

    except ValueError:
        return lf 
    
    if tag == 'sl':
        canonical_value = canonicalize_lf(content)
        return f"[sl:{name} {canonical_value}]"
        
    elif tag == 'in':
        children = _split_children(content)

        canonical_children = [canonicalize_lf(child) for child in children]

        canonical_children.sort()

        joined_children = " ".join(canonical_children)
        return f"[in:{name} {joined_children}]"
        
    else:
        return lf 

def check_correct(target_answer: str, pred_text: str) -> bool:

    canonical_target = canonicalize_lf(target_answer)
    canonical_pred = canonicalize_lf(pred_text)

    return canonical_target == canonical_pred

targets = ['[IN:UPDATE_TIMER [SL:METHOD_TIMER timer ] [SL:TIMER_NAME thaw turkey ] ]','[IN:DELETE_REMINDER [SL:PERSON_REMINDED my ] [SL:TODO [IN:GET_TODO [SL:TODO a playdate ] [SL:ATTENDEE Kim ] ] ] [SL:DATE_TIME tomorrow ] ]','[IN:GET_STORIES_NEWS [SL:NEWS_TYPE headlines ] [SL:NEWS_SOURCE The Guardian ] [SL:DATE_TIME today ] ]']
predictions = ['[IN:CREATE_TIMER [SL:TIMER_NAME Thaw turkey ] [SL:METHOD_TIMER timer ] ]','[IN:DELETE_REMINDER [SL:TODO a playdate with Kim ] [SL:DATE_TIME tomorrow ] ]','[IN:GET_STORIES_NEWS [SL:NEWS_TYPE headlines ] [SL:DATE_TIME today ] [SL:NEWS_SOURCE The Guardian ] ]']

for i in range(len(targets)):
    result = check_correct(targets[i],predictions[i])
    print(result)