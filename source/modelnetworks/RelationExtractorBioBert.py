from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch import nn


class RelationExtractorBioBert(nn.Module):

    def __init__(self, model_dir, num_classes):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained(model_dir,
                                                                   num_labels=num_classes)

        print(self.model)

    def forward(self, input):
        return self.model(input)
