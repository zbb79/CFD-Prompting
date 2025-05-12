import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")

current_dir = Path(__file__).parent
data_path = current_dir.parent / 'models' / 'funing_model'
model = BertModel.from_pretrained(data_path)
tokenizer = BertTokenizer.from_pretrained(data_path)
def encoder(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512,add_special_tokens=True)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)
