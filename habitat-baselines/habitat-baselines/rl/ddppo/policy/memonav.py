import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from typing import List, Optional, Tuple



class TopologicalSTM:
    def __init__(self, stm_capacity: int = 1000, dist_threshold: float = 1.0,memory_hidden_size: int = 1024, ):
        self.stm_capacity = stm_capacity #最大节点数
        self.dist_threshold = dist_threshold #位置去重
        self._memory_hidden_size=memory_hidden_size
        self.reset()

    def reset(self, indices: Optional[List[int]] = None):

        if indices is None:

            self.nodes: List[List[torch.Tensor]] = []
            self.poses: List[List[torch.Tensor]] = []
        else:

            for i in indices:
                if i < len(self.nodes):
                    self.nodes[i] = []
                    self.poses[i] = []


    def _adjust_batch_size(self, B: int):
        current_B = len(self.nodes)
        if B > current_B:
            self.nodes.extend([[] for _ in range(B - current_B)])
            self.poses.extend([[] for _ in range(B - current_B)])
        elif B < current_B:
            self.nodes = self.nodes[:B]
            self.poses = self.poses[:B]

    @torch.no_grad()
    def update(self, visual_feats: torch.Tensor, agent_poses: torch.Tensor):

        B,D = visual_feats.shape
        assert D ==self._memory_hidden_size, f"Input feature dim {D} != expected {self._memory_hidden_size}"

        self._adjust_batch_size(B)

        for b in range(B):
            pose = agent_poses[b].cpu()
            is_new = True

            for p in self.poses[b]:
                if torch.norm(pose[:2] - p[:2]) < self.dist_threshold:
                    is_new = False
                    break

            if is_new and len(self.nodes[b]) < self.stm_capacity:
                self.nodes[b].append(visual_feats[b].detach().detach())
                self.poses[b].append(pose.clone())


    def get_stm_nodes(self, b: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:

        feats = self.nodes[b]
        if not feats:
            return torch.empty(0, self._memory_hidden_size, device=device)
        return torch.stack(feats, dim=0).to(device)

    def get_num_nodes(self, b: int) -> int:
        return len(self.nodes[b])



class StaticLTM(nn.Module):
    def __init__(self, memory_hidden_size: int):
        super().__init__()
        self.ltm = nn.Parameter(torch.randn(1, memory_hidden_size) * 0.02) #可学习的全局记忆节点


    def forward(self, device: torch.device) -> torch.Tensor:
        return self.ltm.to(device)  # [1, D]



def build_memory_graph(stm_feats: torch.Tensor,ltm: torch.Tensor,stm_poses: torch.Tensor,pos_threshold: float = 0.4) -> Data:

    device = ltm.device
    M = stm_feats.size(0)

    if M == 0:
        x = ltm.unsqueeze(0)
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        return Data(x=x, edge_index=edge_index)


    x = torch.cat([stm_feats, ltm], dim=0)
    edges = []

    """
    if M > 1:
        poses_2d = stm_poses[:, :2].to(device)
        dists = torch.cdist(poses_2d, poses_2d)
        rows, cols = torch.where(dists < pos_threshold)
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i != j:
                edges.append((i, j))
    """

    ltm_idx = M
    for i in range(M):
        edges.append((i, ltm_idx))
        edges.append((ltm_idx, i))


    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long,device=device).t().contiguous()  # [2, E]
    else:

        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index)



class ForgettingModule(nn.Module):
    def __init__(self, memory_hidden_size: int, num_heads: int = 4,dropout: float = 0.0):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=memory_hidden_size,
            out_channels=memory_hidden_size,
            heads=num_heads,
            dropout=dropout,
            concat=False,
        )
        self.goal_proj = nn.Linear(memory_hidden_size, memory_hidden_size)

    def forward(
        self,
        data_list: List[Data],
        e_goal: torch.Tensor,
        num_stm_list: List[int],
        forget_ratio: float = 0.8,
        device: torch.device = None,
    ) -> List[torch.Tensor]:

        batch = Batch.from_data_list(data_list).to(device)
        x, edge_index = batch.x, batch.edge_index


        x = F.normalize(x, p=2, dim=-1)
        x_encoded = self.gat(x, edge_index)


        ptr = batch.ptr

        retain_masks = []
        for b in range(len(data_list)):
            M = num_stm_list[b]
            if M == 0:
                retain_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
                continue

            start, end = ptr[b].item(), ptr[b + 1].item()
            stm_repr = x_encoded[start: end - 1]
            stm_repr = F.normalize(stm_repr, p=2, dim=-1)


            goal_emb = self.goal_proj(e_goal[b])
            goal_emb = F.normalize(goal_emb, p=2, dim=-1)

            scores = F.cosine_similarity(goal_emb, stm_repr, dim=-1)
            k = max(1, int(M * forget_ratio))
            topk_idx = torch.topk(scores, k, largest=True).indices
            mask = torch.zeros(M, dtype=torch.bool, device=device)
            mask[topk_idx] = True
            retain_masks.append(mask)
        return retain_masks


class WorkingMemoryEncoder(nn.Module):
    def __init__(self, memory_hidden_size: int, num_heads: int = 4,dropout: float = 0.0):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=memory_hidden_size,
            out_channels=memory_hidden_size,
            heads=num_heads,
            dropout=dropout,
            concat=False,
        )

    def forward(self, data_list: List[Data], device: torch.device) -> Batch:
        batch = Batch.from_data_list(data_list).to(device)
        x, edge_index = batch.x, batch.edge_index
        batch.x = self.gat(x, edge_index)
        return batch


class DualWorkingMemoryDecoder(nn.Module):
    def __init__(self, memory_hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.decoder_cur = nn.MultiheadAttention(
            embed_dim=memory_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_goal = nn.MultiheadAttention(
            embed_dim=memory_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.goal_proj = nn.Linear(memory_hidden_size, memory_hidden_size)
        self.goal_ln = nn.LayerNorm(memory_hidden_size)
        self.norm_cur = nn.LayerNorm(memory_hidden_size)
        self.norm_goal = nn.LayerNorm(memory_hidden_size)

    def forward(
        self,
        visual_feats: torch.Tensor,
        e_goal: torch.Tensor,
        wm_node_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = visual_feats.device
        B = visual_feats.shape[0]
        if B == 0:
            return visual_feats, e_goal
        D = visual_feats.shape[1]
        max_len = max(w.size(0) for w in wm_node_list)
        wm_padded = torch.zeros(B, max_len, D, device=device)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for b, wm in enumerate(wm_node_list):
            L = wm.size(0)
            wm_padded[b, :L] = wm
            key_padding_mask[b, :L] = False

        q_cur = visual_feats.unsqueeze(1)
        q_goal = self.goal_ln(self.goal_proj(e_goal)).unsqueeze(1)

        f_cur, _ = self.decoder_cur(q_cur, wm_padded, wm_padded, key_padding_mask=key_padding_mask)
        f_goal, _ = self.decoder_goal(q_goal, wm_padded, wm_padded, key_padding_mask=key_padding_mask)

        f_cur = self.norm_cur(f_cur.squeeze(1))
        f_goal = self.norm_goal(f_goal.squeeze(1))

        return f_cur.squeeze(1), f_goal.squeeze(1)

class CausalMemoNavMemory(nn.Module):
    def __init__(
        self,
        memory_hidden_size: int =1024,
        stm_capacity: int = 1000,
        dist_threshold: float = 1.0,
        gat_heads: int = 4,
        gat_dropout: float = 0.0,
        train_forget_ratio: float = 0.9,
        decoder_heads: int = 8,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_hidden_size = memory_hidden_size
        self.stm = TopologicalSTM(stm_capacity=stm_capacity,dist_threshold=dist_threshold,memory_hidden_size = memory_hidden_size, )
        self.ltm = StaticLTM(memory_hidden_size)
        self.forgetter = ForgettingModule(memory_hidden_size, gat_heads, gat_dropout)
        self.wm_encoder = WorkingMemoryEncoder(memory_hidden_size, gat_heads,gat_dropout)
        self.wm_decoder = DualWorkingMemoryDecoder(memory_hidden_size,decoder_heads,decoder_dropout)

        self.train_forget_ratio = train_forget_ratio

    def reset(self, indices: Optional[List[int]] = None):

        self.stm.reset(indices)

    def update(self, visual_feats: torch.Tensor, agent_poses: torch.Tensor,dones: torch.Tensor,):


        B = visual_feats.shape[0]
        if B == 0:
            return visual_feats


        if dones.dtype != torch.bool:
            dones = dones.bool()

        reset_indices = dones.nonzero(as_tuple=True)[0].tolist()
        if reset_indices:
            self.reset(reset_indices)


        self.stm.update(visual_feats, agent_poses)

    def forward(
        self,
        visual_feats: torch.Tensor,
        e_goal: torch.Tensor,
        forget_ratio: float = 0.8,
        enable_forget: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = visual_feats.device
        assert e_goal.device == device,

        B = visual_feats.shape[0]
        ltm = self.ltm(device)


        full_data_list = []
        num_stm_list = []
        for b in range(B):
            stm_feats = self.stm.get_stm_nodes(b, device=device)
            M = stm_feats.size(0)
            if M == 0:
                stm_poses = torch.empty(0, 3, device=device)
            else:

                stm_poses = torch.stack(self.stm.poses[b]).to(device) if self.stm.poses[b] else torch.empty(0, 3, device=device)

            data = build_memory_graph(stm_feats, ltm, stm_poses,
                                      pos_threshold=3.0)
            full_data_list.append(data)
            num_stm_list.append(M)

        if enable_forget:

            current_forget_ratio = forget_ratio if not self.training else self.train_forget_ratio
            retain_masks = self.forgetter(
                full_data_list, e_goal, num_stm_list,
                forget_ratio=current_forget_ratio,
                device=device
            )


            pruned_data_list = []
            for b in range(B):
                stm_feats = self.stm.get_stm_nodes(b, device=device)
                M = stm_feats.size(0)
                if M == 0:
                    stm_poses = torch.empty(0, 3, device=device)
                else:
                    stm_poses = torch.stack(self.stm.poses[b], dim=0).to(
                        device)


                stm_feats_pruned = stm_feats[retain_masks[b]]
                stm_poses_pruned = stm_poses[retain_masks[b]]

                pruned_data = build_memory_graph(
                    stm_feats_pruned, ltm, stm_poses_pruned, pos_threshold=3.0
                )
                pruned_data_list.append(pruned_data)
        else:
            pruned_data_list = full_data_list


        wm_batch = self.wm_encoder(pruned_data_list, device)


        final_wm_list: List[torch.Tensor] = []
        ptr = wm_batch.ptr
        for b in range(B):
            nodes = wm_batch.x[ptr[b]:ptr[b + 1]]  # [K_b, D]
            final_wm_list.append(nodes)


        f_cur, f_goal = self.wm_decoder(visual_feats, e_goal, final_wm_list)
        return f_cur, f_goal
