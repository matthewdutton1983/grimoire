def tokenizer(cls):
    tokenizer.tokenizers.append(cls)
    return cls


tokenizer.tokenizers = []
