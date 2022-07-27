from torch.nn import Module
from transformers import AutoConfig, AutoModelForSequenceClassification


class CustomModel(Module):
    def __init__(self, CFG):
        super(CustomModel, self).__init__()
        self.config = AutoConfig.from_pretrained(CFG.model, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CFG.model, num_labels=1
        )

    def forward(self, inputs):
        y = self.model(**inputs).logits
        return y
