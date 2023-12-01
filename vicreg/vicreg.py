from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class VICRegL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = int(args.mlp.split("-")[-1])

        if "convnext" in args.arch:
            import convnext

            self.backbone, self.representation_dim = convnext.__dict__[args.arch](
                drop_path_rate=args.drop_path_rate,
                layer_scale_init_value=args.layer_scale_init_value,
            )
            norm_layer = "layer_norm"
        elif "resnet" in args.arch:
            import resnet

            self.backbone, self.representation_dim = resnet.__dict__[args.arch](
                zero_init_residual=True
            )
            norm_layer = "batch_norm"
        else:
            raise Exception(f"Unsupported backbone {args.arch}.")

        if self.args.alpha < 1.0:
            self.maps_projector = utils.MLP(args.maps_mlp, self.representation_dim, norm_layer)

        if self.args.alpha > 0.0:
            self.projector = utils.MLP(args.mlp, self.representation_dim, norm_layer)

        self.classifier = nn.Linear(self.representation_dim, self.args.num_classes)

    def _vicreg_loss(self, x, y):
        repr_loss = self.args.inv_coeff * F.mse_loss(x, y)

        x = utils.gather_center(x)
        y = utils.gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.args.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.args.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _local_loss(
        self, maps_1, maps_2, location_1, location_2
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.args.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.args.num_matches

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        if self.args.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.args.num_matches[0],
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.args.num_matches[1],
        )

        if self.args.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i], maps_embedding[j], locations[i], locations[j],
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        if self.args.fast_vc_reg:
            inv_loss = self.args.inv_coeff * inv_loss / iter_
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(num_views):
                x = utils.gather_center(maps_embedding[i])
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                x = x.permute(1, 0, 2)
                *_, sample_size, num_channels = x.shape
                non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                x = x - x.mean(dim=-2, keepdim=True)
                cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                cov_loss = cov_loss + cov_loss.mean()
                iter_ = iter_ + 1
            var_loss = self.args.var_coeff * var_loss / iter_
            cov_loss = self.args.cov_coeff * cov_loss / iter_
        else:
            inv_loss = inv_loss / iter_
            var_loss = var_loss / iter_
            cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.args.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = utils.gather_center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + utils.off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.args.var_coeff * var_loss / iter_
        cov_loss = self.args.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_metrics(self, outputs):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                utils.off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = utils.batch_all_gather(outputs["representation"][0])
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.args.alpha > 0.0:
            embedding = utils.batch_all_gather(outputs["embedding"][0])
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
            "logits": [],
            "logits_val": [],
        }
        for x in inputs["views"]:
            maps, representation = self.backbone(x)
            outputs["representation"].append(representation)

            if self.args.alpha > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            if self.args.alpha < 1.0:
                batch_size, num_loc, _ = maps.shape
                maps_embedding = self.maps_projector(maps.flatten(start_dim=0, end_dim=1))
                maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
                outputs["maps_embedding"].append(maps_embedding)

            logits = self.classifier(representation.detach())
            outputs["logits"].append(logits)

        if is_val:
            _, representation = self.backbone(inputs["val_view"])
            val_logits = self.classifier(representation.detach())
            outputs["logits_val"].append(val_logits)

        return outputs

    def forward(self, inputs, is_val=False, backbone_only=False):
        if backbone_only:
            maps, _ = self.backbone(inputs)
            return maps

        outputs = self.forward_networks(inputs, is_val)
        with torch.no_grad():
            logs = self.compute_metrics(outputs)
        loss = 0.0

        # Global criterion
        if self.args.alpha > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(
                outputs["embedding"]
            )
            loss = loss + self.args.alpha * (inv_loss + var_loss + cov_loss)
            logs.update(dict(inv_l=inv_loss, var_l=var_loss, cov_l=cov_loss,))

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.args.alpha < 1.0:
            (
                maps_inv_loss,
                maps_var_loss,
                maps_cov_loss,
            ) = self.local_loss(
                outputs["maps_embedding"], inputs["locations"]
            )
            loss = loss + (1 - self.args.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            logs.update(
                dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
            )

        # Online classification

        labels = inputs["labels"]
        classif_loss = F.cross_entropy(outputs["logits"][0], labels)
        acc1, acc5 = utils.accuracy(outputs["logits"][0], labels, topk=(1, 5))
        loss = loss + classif_loss
        logs.update(dict(cls_l=classif_loss, top1=acc1, top5=acc5, l=loss))
        if is_val:
            classif_loss_val = F.cross_entropy(outputs["logits_val"][0], labels)
            acc1_val, acc5_val = utils.accuracy(
                outputs["logits_val"][0], labels, topk=(1, 5)
            )
            logs.update(
                dict(clsl_val=classif_loss_val, top1_val=acc1_val, top5_val=acc5_val,)
            )

        return loss, logs