# 20200904 Copy from ANCNet CVPR2020

import torch
import torch.nn as nn
import torchvision.models as models
#import pretrainedmodels

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = (
        torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5)
        .unsqueeze(1)
        .expand_as(feature)
    )
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):
    def get_feature_backbone(self, model):
        resnet_feature_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        last_layer = "layer3"
        resnet_module_list = [getattr(model, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        model = nn.Sequential(*resnet_module_list[: last_layer_idx + 1])
        return model

    def __init__(
        self,
        train_fe=False,
        backbone="resnet101",
        feature_extraction_model_file="",
        normalization=True,
        last_layer="",
    ):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn = backbone
        if backbone == "vgg":
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = [
                "conv1_1",
                "relu1_1",
                "conv1_2",
                "relu1_2",
                "pool1",
                "conv2_1",
                "relu2_1",
                "conv2_2",
                "relu2_2",
                "pool2",
                "conv3_1",
                "relu3_1",
                "conv3_2",
                "relu3_2",
                "conv3_3",
                "relu3_3",
                "pool3",
                "conv4_1",
                "relu4_1",
                "conv4_2",
                "relu4_2",
                "conv4_3",
                "relu4_3",
                "pool4",
                "conv5_1",
                "relu5_1",
                "conv5_2",
                "relu5_2",
                "conv5_3",
                "relu5_3",
                "pool5",
            ]
            if last_layer == "":
                last_layer = "pool4"
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(
                *list(self.model.features.children())[: last_layer_idx + 1]
            )
        # for resnet below
        resnet_feature_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        if backbone == "resnet101":
            self.model = models.resnet101(pretrained=True)
            self.model = self.get_feature_backbone(self.model)

        if backbone == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model = self.get_feature_backbone(self.model)

        if backbone == "resnet152":
            model_name = "resnet152"  # could be fbresnet152 or inceptionresnetv2
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = self.get_feature_backbone(model)

        if backbone == "resnet101fcn":
            self.model = gcv.models.get_fcn_resnet101_voc(pretrained=True)

        if backbone == "resnext101":
            model_name = "resnext101_32x4d"  # could be fbresnet152 or inceptionresnetv2
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = model.features[:-1]

        if backbone == "resnext101_64x4d":
            model_name = "resnext101_64x4d"
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = model.features[:-1]

        if backbone == "resnet101fpn":
            if feature_extraction_model_file != "":
                resnet = models.resnet101(pretrained=True)
                # swap stride (2,2) and (1,1) in first layers (PyTorch ResNet is slightly different to caffe2 ResNet)
                # this is required for compatibility with caffe2 models
                resnet.layer2[0].conv1.stride = (2, 2)
                resnet.layer2[0].conv2.stride = (1, 1)
                resnet.layer3[0].conv1.stride = (2, 2)
                resnet.layer3[0].conv2.stride = (1, 1)
                resnet.layer4[0].conv1.stride = (2, 2)
                resnet.layer4[0].conv2.stride = (1, 1)
            else:
                resnet = models.resnet101(pretrained=True)
            resnet_module_list = [getattr(resnet, l) for l in resnet_feature_layers]
            conv_body = nn.Sequential(*resnet_module_list)
            self.model = fpn_body(
                conv_body,
                resnet_feature_layers,
                fpn_layers=["layer1", "layer2", "layer3"],
                normalize=normalization,
                hypercols=True,
            )
            if feature_extraction_model_file != "":
                self.model.load_pretrained_weights(feature_extraction_model_file)

        if backbone == "densenet201":
            self.model = models.densenet201(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])

        if train_fe == False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # # move to GPU
        # self.model = self.model.to(device=device)

    def forward(self, image_batch):
        if self.feature_extraction_cnn == "resnet101fcn":
            features = self.model(image_batch)
            features = torch.cat((features[0], features[1]), 1)
        else:
            features = self.model(image_batch)

        if self.normalization and not self.feature_extraction_cnn == "resnet101fpn":
            features = featureL2Norm(features)

        return features