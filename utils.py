import itertools

from difflib import SequenceMatcher


def is_match(pred: str, gold: str) -> bool:
    pred, gold = pred.strip(), gold.strip()
    if pred == gold:
        return True
    if not pred or not gold:
        return False
    if SequenceMatcher(None, pred, gold).ratio() >= 0.84:
        return True
    return (pred in gold or gold in pred) and abs(len(pred) - len(gold)) <= 5


def is_set_match(pred: str, gold: str, sep_list=("；")) -> bool:
    pred, gold = pred.strip(), gold.strip()
    for sep in sep_list:
        p_items = [s.strip() for s in pred.split(sep) if s.strip()]
        g_items = [s.strip() for s in gold.split(sep) if s.strip()]
        if len(p_items) != len(g_items):
            continue
        if not p_items and not g_items:
            return True
        for perm in itertools.permutations(g_items):
            if all(is_match(p, g) for p, g in zip(p_items, perm)):
                return True
    return False


def cal_em(pred_list, answers, idx=None):
    # ----- 计算 EM -----
    correct = 0
    total = min(len(pred_list), len(answers))
    for i in range(total):
        pred = str(pred_list[i])
        gold = str(answers[i])
        try:
            if is_set_match(pred, gold) or (is_match(pred, gold) and '；' not in gold):
                correct += 1
        except Exception as e:
            print(e)

    em_score = round(correct / total, 4) if total else 0.0
    return em_score


if __name__ == '__main__':
    ans = ['北京',
            '东盟',
            '乔叶',
            '莫斯科不相信眼泪',
            '2016年',
            '新时代、新西藏、新征程——西藏高质量发展与人权保障的新篇章',
            '澳门科技大学',
            '上海松江',
            '黑河',
            '湖南长沙',
            '4',
            '宁波舟山港',
            '朱杨柱',
            '奉节',
            '成都东安湖体育公园主体育场',
            '处在历史十字路口的全球发展',
            '崔宸曦',
            '高志丹',
            '2023年10月10日',
            493,
            '切阳什姐；邓小燕；邓晶；白响恩；吴丹；张晨',
            '中国；柬埔寨；老挝；缅甸；泰国；越南',
            '2014',
            3,
            10]

    pred = ['北京',
            '东盟',
            '乔叶',
            '莫斯科不相信眼泪',
            '2016年',
            '新时代、新西藏、新征程——西藏高质量发展与人权保障的新篇章',
            '香港科技大学',
            '上海松江',
            '黑河',
            '湖南长沙',
            '4',
            '宁波舟山港',
            '朱杨柱',
            '奉节',
            '成都东安湖体育公园主体育场',
            '处在历史十字路口',
            '崔宸曦',
            '高志丹',
            '19年10月10日',
            493,
            '切阳什姐；邓小燕；邓晶；白响恩',
            '中国；柬埔寨；老挝；缅甸；泰国',
            '2014',
            3,
            10]

    print(cal_em(pred, ans))