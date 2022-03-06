import torch
from haystack.modeling.model.language_model import LanguageModel


class WECContextEncoder(LanguageModel):
    def __init__(self):
        super(WECContextEncoder, self).__init__()
        self.model = None
        self.name = "wec_context_encoder"

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, padding_mask: torch.Tensor, **kwargs):
        pass

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass


class WECQuestionEncoder(LanguageModel):
    def __init__(self):
        super(WECQuestionEncoder, self).__init__()
        self.model = None
        self.name = "wec_quesion_encoder"

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor, padding_mask: torch.Tensor, **kwargs):
        pass

    def freeze(self, layers):
        pass

    def unfreeze(self):
        pass
