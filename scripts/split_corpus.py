import json


# {'identifier': 'rmrb_data/20220818/20220818-04-07.txt',
#  'content': '国务院任免国家工作人员\n《\n          人民日报\n          》（\n          2022年08月18日\n\xa0\n          第\xa004\n          版）\n《\n          人民日报\n          》（\n          2022年08月18日\n\xa0\n          第\xa004\n          版）\n新华社北京8月17日电\u3000国务院任免国家工作人员。\n任命孙茂利为公安部副部长；任命王东伟为财政部副部长；任命朱程清（女）为水利部副部长；任命徐加爱为应急管理部副部长；任命潘贤掌为国务院台湾事务办公室副主任。\n免去余蔚平的财政部副部长职务；免去陆桂华的水利部副部长职务。',
#  'file_path': 'rmrb_data/20220818/20220818-04-07.txt',
#  'model_raw_output': '是'}

data = []

start_date = '20230101'
end_date = '20241231'

with open('/home/xiachunxuan/nlp-homework/data/corpus_v3.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        if start_date <= item['file_path'].split('/')[1] <= end_date:
            data.append(item)

# 将筛选后的数据写入新的JSONL文件
with open(f'/home/xiachunxuan/nlp-homework/data/corpus_v3_{start_date}_{end_date}.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
