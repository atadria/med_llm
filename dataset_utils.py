from datasets import load_dataset, Dataset

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
]


def get_text(data_item, label):
    if type(label) == str:
        return {'text': data_item[label]}
    else:
        texts = []
        for d in data_item[label[0]]:
            text_ = d[label[1]]
            assert type(text_) in [str, list]
            if type(text_) == str:
                texts.append(text_)
            elif type(text_) == list:
                texts.append('\n'.join(text_))
        return {'text': '\n'.join(texts)}


def get_dataset():
    multi_data = []

    for dataset in DATASETS:
        data = load_dataset(dataset['id'],
                            split='train',
                            trust_remote_code=True)
        lbl = dataset['label']
        for dp in data:
            multi_data.append(get_text(data_item=dp,
                                       label=lbl))

    return Dataset.from_list(multi_data).shuffle()