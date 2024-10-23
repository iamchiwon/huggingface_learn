from transformers import BertConfig, BertModel

config = BertConfig()
print(config)

model = BertModel(config)

#------

from transformers import BertModel

# download to ~/.cache/huggingface/hub
model = BertModel.from_pretrained("bert-base-cased")

# save to local directory
model.save_pretrained("pretrained-model")
