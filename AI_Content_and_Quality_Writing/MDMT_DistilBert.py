from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import DistilBertModel, DistilBertConfig, DistilBertPreTrainedModel
from transformers.utils import ModelOutput
import torch
from torch.nn import Linear, ModuleList, MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.functional import sigmoid, softmax

@dataclass
class MultiDatasetMultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MDMT_DistilBertConfig(DistilBertConfig):
    def __init__(self, tasks_configs: list[dict] = [], **kwargs):
        super(MDMT_DistilBertConfig, self).__init__(**kwargs)
        self.tasks_configs = tasks_configs

def construct_criterion(name: str):
    if name == 'MSE':
        return MSELoss()
    elif name == 'BCE':
        return BCELoss()
    elif name == 'BCE_Logits':
        return BCEWithLogitsLoss()
    elif name == 'CrossEntropy':
        return CrossEntropyLoss()
    else:
        raise ValueError(f"No implemented loss with name: {name}")

def get_act(loss_name: str):
    if loss_name == 'BCE':
        return sigmoid
    else:
        return (lambda x: x)

class MDMT_DistilBertWrapper(DistilBertPreTrainedModel):
    base_model_prefix = "distilbert"

    def __init__(self, config: MDMT_DistilBertConfig):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.classifiers = ModuleList([Linear(in_features=config.dim, out_features=task_config["target_features"]) for task_config in config.tasks_configs])
        self.criterions = [construct_criterion(task_config["criterion_type"]) for task_config in config.tasks_configs]
        self.activations = [get_act(task_config["criterion_type"]) for task_config in config.tasks_configs]

        for classifier in self.classifiers:
            classifier.reset_parameters()
        self.post_init()

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[list[torch.LongTensor]] = None,
                task_indices: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                loss_weights: Optional[torch.FloatTensor] = None
                ):
        output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_last_hidden_state = output.last_hidden_state[:, 0]

        B = pooled_last_hidden_state.size(0)

        logits = [self.activations[idx](classifier(pooled_last_hidden_state)) for idx, classifier in enumerate(self.classifiers)] # compute logits for all tasks for the whole batch

        loss = None
        if task_indices is not None:
            for i in range(B):
                task_idx = task_indices[i]
                criterion = self.criterions[task_idx]
                task_logits = logits[task_idx][i] # select the right task and then only one example in the batch
                task_labels = labels[i] # select the right example
                task_dataset_weight = loss_weights[task_idx]
                task_width_weight = 1.0/task_labels.size(0)
                task_loss = task_dataset_weight * task_width_weight * criterion(task_logits, task_labels)
                if loss is None:
                    loss = task_loss
                else:
                    loss += task_loss

            loss /= B
        
        logits = torch.cat(logits, dim=1)
        return MultiDatasetMultiTaskModelOutput(logits=logits, loss=loss, attentions=output.attentions, hidden_states=output.hidden_states)