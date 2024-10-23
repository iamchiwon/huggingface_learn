tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

## ---

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
input_ids = tokenizer("Using a Transformer network is simple")
print(input_ids)

tokenizer.save_pretrained("pretrained-tokenizer")

# ---

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# Using a Transformer network is simple
#
# -->
#
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

