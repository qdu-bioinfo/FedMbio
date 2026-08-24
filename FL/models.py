from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn

class MultiModalDiseaseDNN(nn.Module):
    def __init__(self, mic_input_dim):
        super().__init__()

        self.mic_branch = nn.Sequential(
            nn.Linear(mic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        model_path = r"C:\Users\Administrator\Desktop\AAA\distilbert-base-uncased"
        self.bert = DistilBertModel.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.meta_fc = nn.Linear(self.bert.config.hidden_size, 32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, mic_x, meta_texts):
        mic_feat = self.mic_branch(mic_x)
        encoded = self.tokenizer(meta_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(mic_x.device)
        attention_mask = encoded['attention_mask'].to(mic_x.device)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        meta_feat = self.meta_fc(cls_embedding)
        fused = 0.85 * mic_feat + 0.15 * meta_feat
        return fused