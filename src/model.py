from torch import nn
import torch
from src.Unet import AttentionUNet

device = torch.device("cuda:0")
embd_size = 768


class Model(nn.Module):
    def __init__(self, feature_map="similarity"):
        super(Model, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.similarity_3 = nn.Bilinear(embd_size, embd_size, 42, bias=False,device=device)
        self.feature_map = feature_map
        # self.capsule_net = CapsNet()
        self.unet = AttentionUNet(input_channels=3,
                                  class_number=256,
                                  down_channel=256)
        self.unet.to(device)

    def forward(self, x):
        """
        :param x: x is a dictionary which has these keys:
                  embedded_doc, entity_list, label
        :return:
        """
        label = x.get("label")
        entity_list = x["entity_list"]
        entity_list.to(device)
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
        attn_map = nn.Linear(256, 256, device=device)(attn_map)

        stacked_entity = torch.stack([torch.stack([doc] * 42) for doc in entity_list])
        z_s = torch.tanh(nn.Linear(embd_size, 256, bias=False, device = device)(stacked_entity) + attn_map)
        z_o = torch.tanh(nn.Linear(embd_size, 256, bias=False, device = device)(stacked_entity.transpose(1, 2)) + attn_map)
        logits = nn.Bilinear(256, 256, 97, bias=True, device = device)(z_s, z_o)

        return logits
