import re

from datasets import load_dataset, Dataset, concatenate_datasets

# --------------------------------------------------#
# --      text only datasets, no instruction      --#
# --------------------------------------------------#

# use get_dataset function to get all 4 datasets merged into one

DATASETS = [
    {
        'id': 'bigbio/czi_drsm',
        'label': 'text'
    },
    {
        'id': 'bigbio/bc5cdr',
        'label': ['passages', 'text']
    },
    {
        'id': 'bigbio/distemist',
        'label': ['passages', 'text']
    },
    {
        'id': './pubmed/pubmed.py',
        'label': ['MedlineCitation', 'Article', ['ArticleTitle', 'Abstract']]
    }
]


def get_text(data_item, label):
    if type(label) == str:
        return {'text': data_item[label]}
    elif type(label) == list and len(label) == 2:
        texts = []
        for d in data_item[label[0]]:
            text_ = d[label[1]]
            assert type(text_) in [str, list]
            if type(text_) == str:
                texts.append(text_)
            elif type(text_) == list:
                texts.append('\n'.join(text_))
        return {'text': '\n'.join(texts)}
    else:
        # pubmed case
        texts = []
        d = data_item['MedlineCitation']['Article']
        title = d['ArticleTitle']
        abstract = d['Abstract']['AbstractText']
        if abstract:
            texts.append(title)
            if type(abstract) == str:
                texts.append(abstract)
            elif type(abstract) == list:
                texts.append('\n'.join(abstract))
            return {'text': '\n'.join(texts)}


def get_dataset():
    multi_data = []

    for dataset in DATASETS:
        print(f"------ {dataset['id']} ------")
        data = load_dataset(dataset['id'],
                            split='train',
                            trust_remote_code=True)
        lbl = dataset['label']
        for example in data:
            example_text = get_text(data_item=example,
                                    label=lbl)
            if example_text:
                multi_data.append(example_text)
        print(len(multi_data))

    return Dataset.from_list(multi_data).shuffle(seed=123)


# --------------------------------------------------#
# --  instruction datasets - write abstract, QA   --#
# --------------------------------------------------#

# use get_instruction_dataset to get all instruction datasets as text only, aligned with mistral prompt
# * scientific_papers - pubmed subset
# * pubmed_qa - pqa_labeled
# * medmcqa

# scientific_papers
def get_papers_dataset():
    def fix_text(txt):
        txt = txt.replace(' .', '.') \
            .replace(' ,', ',') \
            .replace(' ?', '?') \
            .replace(' )', ')').replace('( ', '(')
        txt = re.sub("(^|[.?!])\s*([a-zA-Z])",
                     lambda p: p.group(0).upper(),
                     txt)
        return txt

    prompt = '''[INST] Write abstract for this article: 
{article}
[/INST]
Abstract:
{abstract} </s>'''

    def format_example(x):
        x['article'] = fix_text(x['article'])
        x['abstract'] = fix_text(x['abstract'])
        x['text'] = prompt.format(**x)
        return x

    data = load_dataset('scientific_papers', 'pubmed',
                        split='train',
                        trust_remote_code=True)
    return data.map(format_example)


# pubmed_qa - 'pqa_artificial', 'pqa_labeled', 'pqa_unlabeled'
def get_pubmed_qa_dataset():
    qa_pubmed_prompt = '''[INST] Provide long answer and short one - final decision (yes/maybe/no) for given question. Use provided context. 
Question: 
{question}

{contexts} [/INST] 
long answer:
{long_answer}

final decision:
{final_decision}</s>'''

    def format_prompt(x):
        contexts = x['context']['contexts']
        context_text = ''
        if len(contexts) == 1:
            context_text = 'context:\n' + contexts[0]
        else:
            for i, c in enumerate(contexts, start=1):
                context_text += f'context {i}:\n{c}\n'
        x['text'] = qa_pubmed_prompt.format(contexts=context_text, **x)
        return x

    data = load_dataset('pubmed_qa', 'pqa_labeled',
                        split='train',
                        trust_remote_code=True)
    print(data)

    return data.map(format_prompt)


# medmcqa
def get_medmcqa_dataset():
    prompt = '''[INST] Select correct option A, B, C, D and provide explanation. 
Topic name: {topic_name_}
Question:
{question}

(A) {opa}
(B) {opb}
(C) {opc}
(D) {opd} [/INST]
Answer:
{cop_label}
Explanation:
{exp} </s>'''

    # skip short answer for questions with choice_type = 'multi'
    #  all questions have only one option selected (cop)
    prompt_multiple = '''[INST] Provide explanation to given question. 
Topic name: {topic_name_}
Question:
{question}

(A) {opa}
(B) {opb}
(C) {opc}
(D) {opd} [/INST]
Explanation:
{exp} </s>'''

    cop_dict = {0: 'A',
                1: 'B',
                2: 'C',
                3: 'D'}

    def format_prompt(x):
        cop_label = cop_dict[x['cop']]
        topic_name_ = x['subject_name']
        if x['topic_name']:
            topic_name_ += ' - ' + x['topic_name']
        if x['choice_type'] == 'multi':
            x['text'] = prompt_multiple.format(cop_label=cop_label,
                                               topic_name_=topic_name_,
                                               **x)
        else:
            x['text'] = prompt.format(cop_label=cop_label,
                                      topic_name_=topic_name_,
                                      **x)
        return x

    data = load_dataset('medmcqa',
                        split='train',
                        trust_remote_code=True)
    return data.map(format_prompt)


def get_instruction_dataset():
    dataset = concatenate_datasets([
        get_papers_dataset().select_columns('text'),
        get_pubmed_qa_dataset().select_columns('text'),
        get_medmcqa_dataset().select_columns('text'),
    ])
    return dataset.shuffle(seed=123)


# --------------------------------------------------#
# --------      Human friendly articles     --------#
# --------------------------------------------------#
