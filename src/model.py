import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.bert.modeling_bert import BertModel


class WecEsModel(nn.Module):
    def __init__(self, tokenizer_len):
        super(WecEsModel, self).__init__()
        self.cfg = BertConfig.from_pretrained("external/spanbert_hf_base")
        self.qa_outputs = nn.Linear(self.cfg.hidden_size, self.cfg.num_labels)

        self.query_spanbert = BertModel.from_pretrained('external/spanbert_hf_base')
        self.query_spanbert.resize_token_embeddings(tokenizer_len)
        self.passage_spanbert = BertModel.from_pretrained('external/spanbert_hf_base')

    def forward(self, passage_input_ids, query_input_ids,
                            passage_input_mask, query_input_mask,
                            passage_segment_ids, query_segment_ids,
                            start_positions, end_positions):
        head_mask = [None] * self.cfg.num_hidden_layers

        passage_encode = self.segment_encode(self.passage_spanbert,
                                             passage_input_ids,
                                             passage_segment_ids,
                                             passage_input_mask,
                                             head_mask)

        query_encode = self.segment_encode(self.query_spanbert,
                                           query_input_ids,
                                           query_segment_ids,
                                           query_input_mask,
                                           head_mask)

        passage_sequence_output = passage_encode[0]
        query_sequence_output = query_encode[0]
        concat_encode_output = torch.cat((passage_sequence_output, query_sequence_output), dim=1)

        # bertlayer_output = self.bert_encoder(concat_encode_output)
        # return self.get_qa_answer(bertlayer_output[0], start_positions, end_positions)
        return self.get_qa_answer(concat_encode_output, start_positions, end_positions)

    @staticmethod
    def segment_encode(model, input_ids, token_type_ids, query_input_mask, head_mask):
        embedding_output = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attention_mask = query_input_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        query_encode = model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return query_encode

    def get_qa_answer(self, encoder_output, start_positions, end_positions):
        logits = self.qa_outputs(encoder_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits
        )
