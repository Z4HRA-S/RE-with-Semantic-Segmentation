from torch import nn
import transformers as ppb
import torch
from src.capsnet import CapsNet
import sys
from src.Unet import AttentionUNet
from memory_profiler import profile

embd_size = 256

class Model(nn.Module):
    def __init__(self, feature_map="similarity"):
        super(Model, self).__init__()
        self.embedding_reduce = torch.nn.Linear(768, embd_size, bias=False)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.similarity_3 = nn.Bilinear(embd_size, embd_size, 42, bias=False)
        self.feature_map = feature_map
        # self.capsule_net = CapsNet()
        self.unet = AttentionUNet(input_channels=3,
                                  class_number=256,
                                  down_channel=256)

    @profile
    def forward(self, x):
        """
        :param x: x is a dictionary which has these keys:
                  embedded_doc, entity_list, label
        :return:
        """
        label = x.pop("label")
        embedded_doc = x["embedded_doc"]
        entity_list = x["entity_list"]
        entity_list = self.embedding_reduce(entity_list)
        # calculating feature_map
        sim_1 = entity_list.matmul(entity_list.transpose(-1, -2)).unsqueeze(-1)
        sim2 = []
        for doc in entity_list:
            sim2_entity_level = []
            for ent in doc:
                sim2_entity_level.append(self.cosine_similarity(ent, doc).unsqueeze(-1))
            sim2.append(torch.concat(sim2_entity_level, dim=-1).unsqueeze(-1))
        sim_2 = torch.concat(sim2, dim=-1).transpose(0, -1).unsqueeze(-1)
        sim_3 = self.similarity_3(entity_list, entity_list).unsqueeze(-1)
        feature_map = torch.concat([sim_1, sim_2, sim_3], dim=-1)  # batch * 42 * 42 * 3
        feature_map = feature_map.transpose(1, 3)  # batch * 3 * 42 * 42
        """output, reconstructions, masked = self.capsule_net(feature_map)
        # take the output and calculate the norm of each 97 vector for each
        vector_norm = torch.Tensor([[torch.norm(vector) for vector in sample] for sample in output])
        predict_index = [torch.argmax(vector) for vector in vector_norm]
        return predict_index"""
        attn_map = self.unet(feature_map)  # ([batch, 42, 42, 256])
        attn_map = nn.Linear(256, 256)(attn_map)
        # final_label = torch.zeros(attn_map.size()[:-1])
        final_label = []
        for batch, doc in enumerate(entity_list):
            for i in range(42):
                for j in range(42):
                    # the entity_list is padded, so we drop the padded info
                    ws_es = torch.nn.Linear(embd_size, 256, bias=False)(doc[i]) + attn_map[batch][i][j]
                    z_s = torch.tanh(ws_es)

                    wo_eo = torch.nn.Linear(embd_size, 256, bias=False)(doc[j]) + attn_map[batch][i][j]
                    z_o = torch.tanh(wo_eo)

                    p_so = torch.nn.Bilinear(256, 256, 97, bias=True)(z_s, z_o)
                    final_label.append(p_so)
                    # p_so = torch.softmax(p_so, dim=0)
                    # final_label[batch][i][j] = torch.argmax(p_so)
        print(final_label[0].size(), final_label[-1].size())
        final_label = torch.stack(final_label)
        print(final_label.size())
        """output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            print("loss", loss.size())
            output = (loss.to(sequence_output), output)"""

        return final_label
