from transformers import BertConfig, T5Config

config = BertConfig.from_pretrained("bert-base-uncased", cache_dir="../.cache/models")
print(config)
config = T5Config.from_pretrained("t5-base", cache_dir="../.cache/models")
print(config)
config = T5Config.from_pretrained("t5-small", cache_dir="../.cache/models")
print(config)
