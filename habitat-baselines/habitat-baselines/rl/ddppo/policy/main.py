
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

from habitat_baselines.rl.ddppo.policy.memonav import CausalMemoNavMemory

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None

@baseline_registry.register_policy
class PointNavResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
        ], f"{backbone} backbone is not recognized."


        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"


        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )


    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):

        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
        )

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()


        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]


        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()


        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )


        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()


        if not self.is_blind:
            spatial_size_h = (
                observation_space.spaces[self.visual_keys[0]].shape[0] // 2
            )
            spatial_size_w = (
                observation_space.spaces[self.visual_keys[0]].shape[1] // 2
            )
            
            self.backbone = make_backbone(
                self._n_input_channels, baseplanes, ngroups
            )



            if hasattr(self.backbone, 'conv1') and self._n_input_channels != 3:
                old_conv = self.backbone.conv1
                self.backbone.conv1 = nn.Conv2d(
                    in_channels=self._n_input_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                    device=old_conv.weight.device,
                    dtype=old_conv.weight.dtype,
                )

                with torch.no_grad():
                    weight = old_conv.weight.clone()
                    new_weight = weight.repeat(1, self._n_input_channels // 3,
                                               1, 1)
                    remainder = self._n_input_channels % 3
                    if remainder:
                        new_weight = torch.cat(
                            [new_weight, weight[:, :remainder]], dim=1)
                    self.backbone.conv1.weight.copy_(new_weight)



            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )

            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )


            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )


            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )


    @property
    def is_blind(self):
        return self._n_input_channels == 0


    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_blind:
            return None


        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]

            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            cnn_input.append(obs_k)


        x = torch.cat(cnn_input, dim=1)

        x = F.avg_pool2d(x, 2)


        x = self.running_mean_and_var(x)

        x = self.backbone(x)

        x = self.compression(x)
        return x


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
    ):
        super().__init__()

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]


        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )


        if not self.is_blind:
            if clip is None:
                raise ImportError(
                    "Need to install CLIP (run `pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656`)"
                )

            model, preprocess = clip.load("RN50")


            self.preprocess = T.Compose(
                [

                    preprocess.transforms[0],
                    preprocess.transforms[1],

                    T.ConvertImageDtype(torch.float),

                    preprocess.transforms[4],
                ]
            )



            self.backbone = model.visual


            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)  # type: Tuple
            elif pooling == "none":
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == "avgpool":
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)


            for param in self.backbone.parameters():
                param.requires_grad = False

            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self._n_input_channels == 0


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None


        cnn_input = []
        if self.rgb:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)


        if self.depth:
            depth_observations = observations["depth"][
                ..., 0
            ]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )
            ddd = torch.stack(
                [
                    self.preprocess(
                        TF.convert_image_dtype(depth_map, torch.uint8)
                    )
                    for depth_map in ddd
                ]
            )
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)


        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x


class PointNavResNetNet(Net):


    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module
    # 初始化参数
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        resnet_baseplanes: int = 32,
        memory_hidden_size: int = 1024,
        causal_gate_threshold: float = 0.4,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()


        self._n_history_frames = 4
        self._rgb_history = None



        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.causal_gate_threshold = causal_gate_threshold
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test


        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()

            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1
        ]
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )


        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32


        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32


        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:

                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)


                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(goal_visual_encoder.output_shape), 1024),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                rnn_input_size += hidden_size

        self._hidden_size = hidden_size


        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict({})
            for k in fuse_keys:
                if len(observation_space.spaces[k].shape) == 3:

                    if k == "rgb":

                        H, W = observation_space.spaces[k].shape[:2]
                        use_obs_space[k] = spaces.Box(
                            low=0, high=255,
                            shape=(H, W, 12),
                            dtype=observation_space.spaces[k].dtype
                        )
                    else:
                        use_obs_space[k] = observation_space.spaces[k]




        if backbone.startswith("resnet50_clip"):
            self.visual_encoder = ResNetCLIPEncoder(
                observation_space
                if not force_blind_policy
                else spaces.Dict({}),
                pooling="avgpool" if "avgpool" in backbone else "attnpool",
            )
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(
                        self.visual_encoder.output_shape[0], hidden_size
                    ),
                    nn.ReLU(True),
                )


        else:
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), 1024),
                    nn.ReLU(True),
                )

            self.global_visual_proj = nn.Sequential(
                nn.Linear(1024 * 3, hidden_size),
                nn.ReLU(True),
                nn.LayerNorm(hidden_size)
            )

            self.goal_proj = nn.Sequential(
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Linear(1024, self._hidden_size)
            )


        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

        self.memory = CausalMemoNavMemory(
            memory_hidden_size=memory_hidden_size,
            # must match visual/goal encoder output
            stm_capacity=1000,
            dist_threshold=1.0,
            gat_heads=4,
            gat_dropout=0.1,
            decoder_heads=8,
            decoder_dropout=0.1,
            train_forget_ratio=0.9,
        )


    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size


    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:


        if not self.is_blind and "rgb" in observations:
            rgb = observations["rgb"]
            B, H, W, C = rgb.shape
            assert C == 3


            if self._rgb_history is None or self._rgb_history[0].shape[0] != B:
                self._rgb_history = [
                    torch.zeros(B, H, W, 3, device=rgb.device, dtype=rgb.dtype)
                    for _ in range(self._n_history_frames)
                ]


            is_new_episode = (masks.squeeze(-1) == 0)

            if is_new_episode.any():

                for i in range(self._n_history_frames):
                    self._rgb_history[i] = torch.where(
                        is_new_episode.view(B, 1, 1, 1),
                        rgb,
                        self._rgb_history[i]
                    )


            self._rgb_history = self._rgb_history[1:] + [rgb]


            observations["rgb"] = torch.cat(self._rgb_history, dim=-1)


        x = []
        aux_loss_state = {}


        if not self.is_blind:

            obs_feat= self.visual_encoder(observations)

            for uuid in [ImageGoalSensor.cls_uuid,InstanceImageGoalSensor.cls_uuid,]:
                if uuid in observations:
                    goal_image = observations[uuid]
                    goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                    goal_feat = goal_visual_encoder({"rgb": goal_image})
                    goal_visual_fc = getattr(self, f"{uuid}_fc")
                    e_goal = goal_visual_fc(goal_feat)
                    e_goal = F.normalize(e_goal, dim=-1)



            obs_vec = F.adaptive_avg_pool2d(obs_feat, (1, 1)).flatten(1)
            goal_vec = F.adaptive_avg_pool2d(goal_feat, (1, 1)).flatten(1)


            obs_norm = F.normalize(obs_vec, dim=1)
            goal_norm = F.normalize(goal_vec, dim=1)
            similarity = torch.sum(obs_norm * goal_norm, dim=1)

            use_causal = similarity > self.causal_gate_threshold
            normal_feat = self.visual_fc(obs_feat)
            normal_out = F.normalize(normal_feat, dim=-1)

            if use_causal.any():

                B, D, H, W = obs_feat.shape
                obs_flat = obs_feat.flatten(2).permute(0, 2, 1)
                goal_flat = goal_feat.flatten(2).permute(0, 2, 1)


                attn_is = torch.bmm(obs_flat, goal_flat.transpose(1, 2))
                attn_is = F.softmax(attn_is / (D ** 0.5), dim=-1)
                Z_hat = torch.bmm(attn_is, goal_flat)


                attn_cs = torch.bmm(obs_flat, obs_flat.transpose(1, 2))
                attn_cs = F.softmax(attn_cs / (D ** 0.5), dim=-1)
                X_hat = torch.bmm(attn_cs, obs_flat)

                causal_feat = (Z_hat + X_hat).permute(0, 2, 1)
                causal_feat = self.visual_fc(causal_feat)
                causal_out = F.normalize(causal_feat, dim=-1)
            else:
                causal_out = normal_out



            visual_feats = torch.where(
                use_causal.unsqueeze(1),
                causal_out,
                normal_out
            )



        device =visual_feats.device


        if EpisodicGPSSensor.cls_uuid not in observations or EpisodicCompassSensor.cls_uuid not in observations:
            raise RuntimeError(
                "CausalMemoNavMemory requires EPISODIC_GPS and EPISODIC_COMPASS sensors.")
        gps = observations[EpisodicGPSSensor.cls_uuid].float()
        compass = observations[EpisodicCompassSensor.cls_uuid].float()

        agent_poses = torch.cat([gps, compass], dim=-1)


        dones = (~masks.squeeze(-1)).bool()
        with torch.no_grad():
            self.memory.update(visual_feats.detach(), agent_poses, dones)


        f_cur, f_goal = self.memory(
            visual_feats=visual_feats,
            e_goal=e_goal,
            enable_forget=True
        )

        fused_visual = torch.cat([visual_feats,f_cur,f_goal],dim=-1)


        visual_feats = self.global_visual_proj(fused_visual)
        e_goal_proj = self.goal_proj(e_goal)
        aux_loss_state["perception_embed"] = visual_feats
        x.append(visual_feats)
        x.append(e_goal_proj)



        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())


        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )


        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)

            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)


        out = torch.cat(x, dim=1)

        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state
