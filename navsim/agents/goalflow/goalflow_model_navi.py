import math
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.goalflow.goalflow_config import GoalFlowConfig
from navsim.common.enums import StateSE2Index
from navsim.agents.goalflow.goalflow_features import BoundingBox2DIndex
from navsim.agents.goalflow.utils import pos2posemb2d
from navsim.agents.goalflow.v99_backbone import V299Backbone

import torch.nn.functional as F


class GoalFlowNaviModel(nn.Module):
    def __init__(self, config: GoalFlowConfig):

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ] # target trajs query + bboxes query

        self._config = config
        self._backbone=V299Backbone(config) # V299Backbone: fusion camera_feature + lidar_feature
        
        # load Goal Point Vocabulary
        if not self._config.voc_path=='':
            self.cluster_points=np.load(self._config.voc_path)
        
        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)  # 特征降维

        # BEV semantic segmentation head
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._keyval_embedding2 = nn.Embedding(
            8**2 , config.tf_d_model
        )  # 8x8 feature grid + trajectory
        
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        
        self._status_encoding=nn.Linear(4+2+2,config.tf_d_model)
        # self._status_encoding=nn.Linear((4+3+2+2)*4,config.tf_d_model)

        # Transformer decoder解码器层
        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,  # 256
            nhead=config.tf_num_head,   # 8
            dim_feedforward=config.tf_d_ffn,    # 1024
            dropout=config.tf_dropout,  # 0.0
            batch_first=True,
        )

        self._tf_decoder2 = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._voc_decoder = nn.TransformerDecoder(tf_decoder_layer,config.tf_num_layers)
        
        self.voc_mlp=nn.Sequential(
            nn.Linear(3,self._config.tf_d_model),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model,self._config.tf_d_model)
        )
        # IM Score MLP: 预测模仿学习分数(衡量目标点与真实轨迹终点接近程度)
        self._im_mlp=nn.Sequential(
            nn.Linear(self._config.tf_d_model,self._config.tf_d_model//2),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model//2,1)
        )
        # DAC Score MLP: 预测领域适应分数(衡量目标点是否在可行驶区域内)
        self._dac_mlp=nn.Sequential(
            nn.Linear(self._config.tf_d_model,self._config.tf_d_model//2),
            nn.ReLU(),
            nn.Linear(self._config.tf_d_model//2,1) # 2分类
        )


        if self._config.freeze_perception: # freeze perception backbone
            self.freeze_layers(self._backbone)
            self.freeze_layers(self._bev_downscale)
            self.freeze_layers(self._bev_semantic_head)
            self.freeze_layers(self._query_embedding)
            self.freeze_layers(self._status_encoding)


    def freeze_layers(self, layer):
        for param in layer.parameters():
            param.requires_grad = False


    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # 1.Extract features(输入特征提取)
        token=features['token']
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        if self._config.training:
            dac_score_feature: torch.Tensor = features["dac_score_feature"]

        # =================================== perception  ==================================================
        batch_size = status_feature.shape[0]
        
        # bev_feature_upscale(bz,64,64,64), bev_feature(bz,512,8,8)
        # upscale for bev semantic segmentation head(高分辨率用于语义分割,低分辨率用于transformer decoder)
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        # bev_feature(bz,64,256)
        gt_trajs=features['gt_trajs'].to(status_feature)


        # =================================== goal point decoder ==================================================
        # 2.Status Encoding
        if self._config.has_history:
            used_dims=[0,1,2,3,7,8,9,10] # select dims from status_feature
            status_encoding = self._status_encoding(status_feature[:,-1,used_dims])
        else:
            status_encoding=self._status_encoding(status_feature)
        # status_encoding (1,256)
        
        # 3.Goal Point Vocabulary Decoder
        cluster_points_tensor=torch.from_numpy(self.cluster_points)[None,...].repeat(batch_size,1,1).to(gt_trajs)
        # cluster_points_embed=self.voc_mlp(cluster_points_tensor)
        cluster_points_embed=pos2posemb2d(cluster_points_tensor,self._config.tf_d_model//2)
        
        # 4.Query and Key-Value for Transformer Decoder
        query=self._voc_decoder(cluster_points_embed,cluster_points_embed)+status_encoding[:,None] # 目标点词汇+自车状态
        
        # keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval = bev_feature+self._keyval_embedding2.weight[None, ...] # BEV特征+位置嵌入

        # query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder2(query, keyval) # Query attend to BEV features
        # query_out (bz,8192,256)
        
        im_scores=self._im_mlp(query_out)
        dac_scores=self._dac_mlp(query_out)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        gt_navi=gt_trajs[:,7:8,:]
        if self._config.training:
            im_scores_loss=imitation_loss(gt_navi,cluster_points_tensor,im_scores) # cal im loss
            dac_scores_loss=dac_loss(dac_scores,dac_score_feature)  # cal dac loss

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        if self._config.training:
            output.update({'im_score_loss':im_scores_loss})
            output.update({'dac_score_loss':dac_scores_loss})
        else:
            output.update({'im_scores':im_scores})
            output.update({'dac_scores':dac_scores})

        return output


def imitation_loss(T_hat, T, S_im): 
    # calculate imitation loss: distance between vocabulary points and ground truth
    distances = torch.sum((T_hat - T) ** 2, dim=-1)

    y_i = F.softmax(-distances, dim=-1)
    S_im = F.softmax(S_im.squeeze(), dim=-1)

    loss = -torch.sum(y_i * torch.log(S_im+1e-6), dim=-1)   # Cross-entropy loss

    return loss.mean()

def dac_loss(S_hat, S):
    # calculate dac loss: 1/0 loss for whether the vocabulary points are in drivable area
    S_prob=torch.sigmoid(S_hat.squeeze())
    loss = - (S * torch.log(S_prob + 1e-6) + (1 - S) * torch.log(1 - S_prob + 1e-6))

    total_loss = loss.sum(dim=1)

    return total_loss.mean()