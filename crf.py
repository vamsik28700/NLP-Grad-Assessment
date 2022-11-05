#import statements
import torch
from torch import nn

# Defined a simple but work in progress CRF Module
class CRF(nn.Module):

    def __init__(
        self, nb_labels, bos_tag_id, eos_tag_id, pad_tag_id=None, batch_first=True
    ):
        super().__init__()

        self.nb_labels = nb_labels
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.PAD_TAG_ID = pad_tag_id
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        # initializing random uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        
        self.transitions.data[:, self.BOS_TAG_ID] = -10000.0
        self.transitions.data[self.EOS_TAG_ID, :] = -10000.0

        if self.PAD_TAG_ID is not None:
            self.transitions.data[self.PAD_TAG_ID, :] = -10000.0
            self.transitions.data[:, self.PAD_TAG_ID] = -10000.0
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0
