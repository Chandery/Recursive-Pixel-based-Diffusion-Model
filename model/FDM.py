import torch
import lightning as L
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from .DiT import DiT
from diffusion import create_diffusion

class NeighborEmbedder(nn.Module): # ! Key Innovation point
    """Process neighboring patches using Self-Attention and Cross-Attention"""
    def __init__(self, patch_size, embed_dim, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embedder = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 9, embed_dim))
        
        # Self-Attention: let neighbors interact with each other
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-Attention: center patch extracts information from neighbors
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
    def forward(self, neighbor_patches):
        """
        neighbor_patches: [batch, num_h, num_w, 9, 3, patch_size, patch_size]
        
        9 positions layout:
        [0] [1] [2]    top-left    top    top-right
        [3] [4] [5]    left      center   right
        [6] [7] [8]    bottom-left bottom bottom-right
        """
        batch_size, num_h, num_w, num_neighbors, C, ps, _ = neighbor_patches.shape
        
        # ============ Data Preparation ============
        # Reshape to [batch*num_patches, 9, 3*ps*ps]
        x = neighbor_patches.view(
            batch_size * num_h * num_w, num_neighbors, C * ps * ps
        )
        # Now x[i, j, :] represents the j-th neighbor of the i-th patch (flattened RGB pixels)
        
        # Embed: [B*N, 9, 3*ps*ps] → [B*N, 9, D]
        x = self.patch_embedder(x)
        
        # Add positional encoding
        x = x + self.pos_embed  # [B*N, 9, D] + [B*N, 9, D]
        
        # * ============ Step 1: Self-Attention ============
        # Let 9 neighbors interact with each other
        # Q = x @ W_q  # [B*N, 9, D] @ [D, D] = [B*N, 9, D]
        # K = x @ W_k  # [B*N, 9, D] @ [D, D] = [B*N, 9, D]
        # V = x @ W_v  # [B*N, 9, D] @ [D, D] = [B*N, 9, D]
        # scores = Q @ K^T  # [B*N, 9, D] @ [B*N, D, 9] = [B*N, 9, 9]
        # attn_weights = softmax(scores)  # [B*N, 9, 9]
        # attn_out = attn_weights @ V  # [B*N, 9, 9] @ [B*N, 9, D] = [B*N, 9, D]
        
        attn_out, _ = self.self_attn(
            query=x,  # [B*N, 9, D]
            key=x,    # [B*N, 9, D]
            value=x   # [B*N, 9, D]
        )
        # attn_out: [B*N, 9, D]
        
        # Residual connection + LayerNorm
        x = self.norm1(x + attn_out)  # [B*N, 9, D]
        
        # * ============ Step 2: Cross-Attention ============
        # * Center patch extracts information from neighbors
        center_query = x[:, 4:5, :]  # [B*N, 1, D] - Extract center patch (index=4)
        
        # Q = center_query @ W_q  # [B*N, 1, D] @ [D, D] = [B*N, 1, D]
        # K = x @ W_k             # [B*N, 9, D] @ [D, D] = [B*N, 9, D]
        # V = x @ W_v             # [B*N, 9, D] @ [D, D] = [B*N, 9, D]
        # scores = Q @ K^T        # [B*N, 1, D] @ [B*N, D, 9] = [B*N, 1, 9]
        # attn_weights = softmax(scores)  # [B*N, 1, 9]
        # attn_out = attn_weights @ V     # [B*N, 1, 9] @ [B*N, 9, D] = [B*N, 1, D]
        
        attn_out, _ = self.cross_attn(
            query=center_query,  # [B*N, 1, D] - Center patch
            key=x,              # [B*N, 9, D] - All neighbors
            value=x
        )
        # attn_out: [B*N, 1, D]
        
        # Residual connection + LayerNorm
        center_out = self.norm2(center_query + attn_out)  # [B*N, 1, D]
        
        ffn_out = self.ffn(center_out)  # [B*N, 1, D] → [B*N, 1, 4D] → [B*N, 1, D]
        output = self.norm3(center_out + ffn_out)
        
        output = output.squeeze(1).view(batch_size, num_h * num_w, self.embed_dim)
        # [batch, num_h*num_w, embed_dim]
        
        return output

class FractalDiT_Module(nn.Module):
    def __init__(self,
                 img_size_list,
                 patch_size_list,
                 embed_dim_list,
                 num_head_list,
                 depth_list,
                 timestep_list,
                 class_num,
                 fractal_level=0,
                 max_patches_per_step=256,
                 *args,
                 **kwargs):
        super().__init__()
        
        # * fractal configure
        self.fractal_level = fractal_level
        self.fractal_level_total = len(img_size_list)
        
        # * diffusion configure
        self.img_size = img_size_list[fractal_level]
        self.patch_size = patch_size_list[fractal_level]
        self.embed_dim = embed_dim_list[fractal_level]
        self.num_heads = num_head_list[fractal_level]
        self.depth = depth_list[fractal_level]
        self.timesteps = timestep_list[fractal_level]
        self.class_num = class_num
        
        # * diffusion setup
        self.DiT_model = DiT(
            input_size=self.img_size,
            depth=self.depth,
            in_channels=3,
            hidden_size=self.embed_dim,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            num_classes=class_num
        )
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.timesteps
        )
        
        # * neighbor processing
        self.neighbor_embedder = NeighborEmbedder(self.img_size, self.embed_dim)
        
        # * layer for fusing neighbor_cond into the model
        self.neighbor_fusion = nn.Linear(self.embed_dim, self.embed_dim)
        
        # * position encoding for first-level patches (before DiT)
        # 使用可学习的线性层将位置编码 (i, j) 映射到 embed_dim
        self.patch_pos_embed = nn.Linear(2, self.embed_dim)  # (i, j) -> embed_dim
        
        # * gradient checkpointing 配置
        # checkpointing 会显著增加反向传播时间（需要重新计算激活值）
        # 但可以节省显存。建议：如果显存够用，关闭以加速训练
        self.use_checkpoint_neighbor = False  # neighbor_embedder 通常不需要 checkpointing
        self.use_checkpoint_dit = False  # DiT blocks 的 checkpointing，如果显存不够可以开启
        
        # * 随机采样配置（用于减少计算量）
        # 当 patch 数量很大时，随机采样一部分 patches 来训练，而不是全部
        # 这样可以显著减少计算量，同时保持训练效果
        self.max_patches_per_step = max_patches_per_step  # 每次随机采样 2048 个 patches 进行训练
    
    def _neighbor_embedder_forward(self, neighbor_patches):
        """包装函数用于 gradient checkpointing"""
        return self.neighbor_embedder(neighbor_patches)
    
    def _dit_block_forward(self, block, x, c):
        """包装函数用于 gradient checkpointing"""
        return block(x, c)
    
    def model_wrapper(self, x, t, y, neighbor_cond=None, patch_pos=None):
        """
        这个包装器会被diffusion.training_losses调用
        它接受model_kwargs中的neighbor_cond和patch_pos参数
        x: [batch, 3, patch_size, patch_size] - 单个patch（第一个patch阶段）
        需要上采样到 [batch, 3, img_size, img_size] 以满足 x_embedder 的输入要求
        patch_pos: [batch, 2] - patch的位置编码 (i, j) 坐标，归一化到 [0, 1]
        """
        # 如果输入是单个patch，需要上采样到完整图像大小
        if x.shape[-1] == self.patch_size and x.shape[-1] != self.img_size:
            # 使用双线性插值上采样
            x = torch.nn.functional.interpolate(
                x, 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 先通过DiT获取基础的patch embeddings
        x_embed = self.DiT_model.x_embedder(x)
        # print(f"[DEBUG] x_embed shape: {x_embed.shape}; pos_embed shape: {self.DiT_model.pos_embed.shape}")
        x_embed = x_embed + self.DiT_model.pos_embed  # (N, T, D) - DiT 内部的位置编码
        
        # 添加第一个 patch 阶段的位置编码
        if patch_pos is not None:
            # patch_pos: [batch, 2] - (i, j) 坐标归一化到 [0, 1]
            # 通过线性层映射到 embed_dim
            patch_pos_embed = self.patch_pos_embed(patch_pos)  # [batch, embed_dim]
            # x_embed: [batch, num_patches, embed_dim]，需要将 patch_pos_embed 扩展到所有 patches
            num_patches = x_embed.shape[1]
            patch_pos_embed_expanded = patch_pos_embed.unsqueeze(1).expand(-1, num_patches, -1)  # [batch, num_patches, embed_dim]
            x_embed = x_embed + patch_pos_embed_expanded  # 添加到所有 patches
        
        t_embed = self.DiT_model.t_embedder(t)  # (N, D)
        y_embed = self.DiT_model.y_embedder(y, self.DiT_model.training)  # (N, D)
        
        # if neighbor_cond, fuse it in
        if neighbor_cond is not None:
            # neighbor_cond: [batch, 1, embed_dim] (单个patch的neighbor condition)
            # x_embed: [batch, num_patches, embed_dim] (完整图像的patch embeddings)
            # 需要将neighbor_cond扩展到所有patches，或者只添加到对应的patch
            if neighbor_cond.dim() == 3 and neighbor_cond.shape[1] == 1:
                # 将neighbor_cond扩展到所有patches
                num_patches = x_embed.shape[1]
                neighbor_cond_expanded = neighbor_cond.expand(-1, num_patches, -1)
            else:
                neighbor_cond_expanded = neighbor_cond
            neighbor_info = self.neighbor_fusion(neighbor_cond_expanded)
            x_embed = x_embed + neighbor_info  # simple addition fusion
        
        # continue DiT's normal process
        c = t_embed + y_embed  # (N, D)
        
        # 使用 gradient checkpointing 减少显存占用（会显著增加反向传播时间）
        # 检查模型是否处于训练模式
        is_training = self.DiT_model.training if hasattr(self.DiT_model, 'training') else True
        if self.use_checkpoint_dit and is_training:
            # 对 DiT blocks 使用 checkpointing（会显著增加反向传播时间）
            for block in self.DiT_model.blocks:
                x_embed = checkpoint(self._dit_block_forward, block, x_embed, c, use_reentrant=False)
        else:
            # 正常前向传播（推荐：如果显存够用）
            for block in self.DiT_model.blocks:
                x_embed = block(x_embed, c)  # (N, T, D)
        
        x_embed = self.DiT_model.final_layer(x_embed, c)  # (N, T, patch_size ** 2 * out_channels)
        x_out = self.DiT_model.unpatchify(x_embed)  # (N, out_channels, H, W)
        
        return x_out
            
    def forward(self, x, t, y, neighbor_cond=None, patch_pos=None):
        """
        x: [batch, 3, H, W]
        t: [batch]
        y: [batch] label
        neighbor_cond: [batch, num_patches, embed_dim] optional neighbor condition
        patch_pos: [batch, 2] optional patch position encoding (i, j) normalized to [0, 1]
        """
        return self.model_wrapper(x, t, y, neighbor_cond, patch_pos)
    
    def forward_with_cfg(self, x, t, y, cfg_scale, neighbor_cond=None):
        # use wrapper to support cfg for neighbor_cond
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        if neighbor_cond is not None:
            # copy neighbor_cond
            half_neighbor = neighbor_cond[: len(neighbor_cond) // 2]
            combined_neighbor = torch.cat([half_neighbor, half_neighbor], dim=0)
        else:
            combined_neighbor = None
            
        model_out = self.model_wrapper(combined, t, y, combined_neighbor)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class FDM(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.img_size_list = config["FDM"].img_size_list
        self.timestep_list = config["FDM"].timestep_list
        self.total_stages = len(self.img_size_list)
        
        self.lr = config.lr
        
        for stage in range(self.total_stages):
            self.add_module(
                f"model_stage_{stage}", 
                FractalDiT_Module(**config["FDM"], fractal_level=stage)
            )
    
    def patchify(self, x, patch_size):
        """
        Split the image into patches
        x: [batch, 3, H, W]
        output: [batch, num_patches_h, num_patches_w, 3, patch_size, patch_size]
        """
        batch_size, channels, height, width = x.shape
        assert height % patch_size == 0 and width % patch_size == 0
        
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Reshape to patches
        patches = x.view(batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        # [batch, num_patches_h, num_patches_w, channels, patch_size, patch_size]
        
        return patches
    
    def get_neighbor_patches(self, patches):
        """
        Get the 8 neighbors of each patch (including itself)
        patches: [batch, num_patches_h, num_patches_w, 3, patch_size, patch_size]
        output: [batch, num_patches_h, num_patches_w, 9, 3, patch_size, patch_size]
        
        优化：使用向量化操作替代 for 循环，大幅加速
        """
        batch_size, num_h, num_w, channels, ps, _ = patches.shape
        
        # Pad patches to get boundary neighbors
        # Rearrange to [batch, 3, num_h, num_w, patch_size, patch_size]
        patches_padded = patches.permute(0, 3, 1, 2, 4, 5)
        # Reshape to [batch, 3, num_h*patch_size, num_w*patch_size]
        patches_padded = patches_padded.contiguous().view(
            batch_size, channels, num_h * ps, num_w * ps
        )
        
        # Pad image boundaries
        padded = torch.nn.functional.pad(patches_padded, (ps, ps, ps, ps), mode='reflect')
        
        # Repatchify padded image
        padded_patches = self.patchify(padded, ps)
        # [batch, num_h+2, num_w+2, 3, patch_size, patch_size]
        padded_patches_perm = padded_patches.permute(0, 3, 4, 5, 1, 2).contiguous()
        # [batch, 3, ps, ps, num_h+2, num_w+2]
        
        # 使用 unfold 在最后两个维度（空间维度）上提取 3x3 窗口
        # unfold(dim, size, step) 在指定维度上提取滑动窗口
        unfolded_h = padded_patches_perm.unfold(4, 3, 1)  # 在 num_h+2 维度上提取 3x1 窗口
        # [batch, 3, ps, ps, num_h, num_w+2, 3]
        unfolded = unfolded_h.unfold(5, 3, 1)  # 在 num_w+2 维度上提取 3x1 窗口
        # [batch, 3, ps, ps, num_h, num_w, 3, 3]
        
        # 重新排列为 [batch, num_h, num_w, 3, 3, 3, ps, ps]
        unfolded = unfolded.permute(0, 4, 5, 1, 6, 7, 2, 3).contiguous()
        # [batch, num_h, num_w, 3, 3, 3, ps, ps]
        all_neighbors = unfolded.view(batch_size, num_h, num_w, 9, channels, ps, ps)
        
        return all_neighbors
    
    def unpatchify(self, patches, patch_size):
        """
        Reassemble patches into an image
        patches: [batch, num_patches_h, num_patches_w, 3, patch_size, patch_size]
        output: [batch, 3, H, W]
        """
        batch_size, num_h, num_w, channels, ps, _ = patches.shape

        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous() # ? should be contiguous in the RAM before view
        # [batch, channels, num_h, patch_size, num_w, patch_size]
        img = patches.view(batch_size, channels, num_h * ps, num_w * ps)
        
        return img
    
    def training_step(self, batch, batch_idx):
        """
        Reverse Order Training: Stage 2 (1x1) → 1 (4x4) → 0 (16x16)...
        Cascaded Noising: Stage 2 (1x1) → 1 (4x4) → 0 (16x16)...
        
        使用渐进式 Teacher Forcing：
        - 训练初期：更多使用干净输入（teacher forcing）
        - 训练后期：更多使用真实场景（前一个 stage 的输出）
        """
        x, y = batch
        x_original = x.clone()  # 保存原始图像用于 teacher forcing
        x_cascaded = x.clone()  # 维护累积加噪状态（用于保持正确的加噪步数）
        total_loss = 0
        
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 1000
        
        progress = min(current_epoch / max_epochs, 1.0)
        teacher_forcing_prob = 0.5 * (1.0 + math.cos(math.pi * progress))  # 从 1.0 到 0.0
        
        if batch_idx == 0:  # 每个 epoch 记录一次
            self.log("teacher_forcing_prob", teacher_forcing_prob, on_step=False, on_epoch=True)
        
        for stage in range(self.total_stages - 1, -1, -1):
            current_stage_model = getattr(self, f"model_stage_{stage}")
            patch_size = self.img_size_list[stage]
            
            # patches: [batch, num_h, num_w, 3, patch_size, patch_size]
            patches = self.patchify(x, patch_size)
            batch_size, num_h, num_w, channels, ps, _ = patches.shape
            
            # neighbor_patches: [batch, num_h, num_w, 9, 3, patch_size, patch_size]
            neighbor_patches = self.get_neighbor_patches(patches)
            
            # neighbor_cond: [batch, num_h*num_w, embed_dim]
            # neighbor_embedder 通常不需要 checkpointing（计算量小，显存占用不大）
            if current_stage_model.use_checkpoint_neighbor and self.training:
                neighbor_cond = checkpoint(
                    current_stage_model._neighbor_embedder_forward, 
                    neighbor_patches, 
                    use_reentrant=False
                )
            else:
                neighbor_cond = current_stage_model.neighbor_embedder(neighbor_patches)
            
            # [batch, num_h, num_w, 3, ps, ps] → [batch*num_h*num_w, 3, ps, ps]
            patches_flat = patches.view(batch_size * num_h * num_w, channels, ps, ps)
            num_patches = batch_size * num_h * num_w
            
            # 生成位置编码：每个 patch 的 (i, j) 坐标，归一化到 [0, 1]
            # i: [0, num_h-1], j: [0, num_w-1]
            i_coords = torch.arange(num_h, device=x.device).float() / (num_h - 1) if num_h > 1 else torch.zeros(1, device=x.device)
            j_coords = torch.arange(num_w, device=x.device).float() / (num_w - 1) if num_w > 1 else torch.zeros(1, device=x.device)
            # 生成所有 (i, j) 组合
            i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
            # [num_h, num_w] -> [num_h*num_w, 2]
            patch_positions = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=1)
            # 扩展到 batch: [batch*num_h*num_w, 2]
            patch_positions = patch_positions.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(-1, 2)
            
            # 随机采样策略：只训练一部分 patches 以减少计算量
            max_patches = current_stage_model.max_patches_per_step
            indices = None
            if  max_patches is not None and num_patches > max_patches:
                # 随机采样 max_patches 个 patches
                indices = torch.randperm(num_patches, device=x.device)[:max_patches]
                patches_flat = patches_flat[indices]
                num_patches = max_patches
                # 也需要采样对应的 neighbor_cond 和位置编码
                neighbor_cond_flat = neighbor_cond.view(batch_size * num_h * num_w, 1, -1)[indices]
                patch_positions = patch_positions[indices]
            else:
                # neighbor_cond: [batch, num_h*num_w, embed_dim] → [batch*num_h*num_w, 1, embed_dim]
                neighbor_cond_flat = neighbor_cond.view(num_patches, 1, -1)
            
            # print("[DEBUG] patches_flat shape: ", patches_flat.shape)
            
            t = torch.randint(
                0, self.timestep_list[stage], 
                (num_patches,),  # ! each patch has a time step (Key Innovation point)
                device=x.device
            )
            
            # y: [batch] → [batch*num_h*num_w] (如果采样了，需要重新计算)
            y_expanded = y.unsqueeze(1).expand(batch_size, num_h * num_w).contiguous().view(-1)
            
            # if sampled, only use the sampled patches
            if indices is not None:
                y_expanded = y_expanded[indices]
            
            # 直接处理采样后的 patches
            model_kwargs = {
                "y": y_expanded,
                "neighbor_cond": neighbor_cond_flat,
                "patch_pos": patch_positions  # [num_patches, 2] - 位置编码 (i, j) 归一化到 [0, 1]
            }
            loss_dict = current_stage_model.diffusion.training_losses(
                model=current_stage_model.model_wrapper,
                x_start=patches_flat,  # [num_patches, 3, patch_size, patch_size]
                t=t,
                model_kwargs=model_kwargs
            )
            loss = loss_dict["loss"].mean()
            
            total_loss += loss
            
            self.log(f"train_loss_stage_{stage}", loss, prog_bar=True)
            
            # print("[DEBUG] Before Cascaded Noising")
            
            # Cascaded Noising: for next stage
            # 使用渐进式 Teacher Forcing：部分时间使用原始干净图像，部分时间使用累积加噪后的图像
            # 关键：x_cascaded 维护累积加噪状态，确保加噪步数连续正确
            if stage > 0:
                with torch.no_grad():
                    noise_t = torch.full(
                        (batch_size,), 
                        self.timestep_list[stage] - 1,
                        device=x_cascaded.device,
                        dtype=torch.long
                    )
                    noise = torch.randn_like(x_cascaded)
                    x_cascaded = current_stage_model.diffusion.q_sample(x_cascaded, noise_t, noise=noise)
                    
                    use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
                    
                    if use_teacher_forcing:
                        x = x_original.clone()
                        if batch_idx == 0:
                            self.log(f"teacher_forcing_stage_{stage}", 1.0, on_step=False, on_epoch=True)
                    else:
                        x = x_cascaded.clone()
                        if batch_idx == 0:
                            self.log(f"teacher_forcing_stage_{stage}", 0.0, on_step=False, on_epoch=True)

            # print("[DEBUG] After Cascaded Noising")
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step: compute validation loss for each stage
        Similar to training but without gradient updates
        """
        x, y = batch
        total_loss = 0
        
        for stage in range(self.total_stages - 1, -1, -1):
            current_stage_model = getattr(self, f"model_stage_{stage}")
            patch_size = self.img_size_list[stage]
            
            # patches: [batch, num_h, num_w, 3, patch_size, patch_size]
            patches = self.patchify(x, patch_size)
            batch_size, num_h, num_w, channels, ps, _ = patches.shape
            
            # neighbor_patches: [batch, num_h, num_w, 9, 3, patch_size, patch_size]
            neighbor_patches = self.get_neighbor_patches(patches)
            
            # neighbor_cond: [batch, num_h*num_w, embed_dim]
            # neighbor_embedder 通常不需要 checkpointing（计算量小，显存占用不大）
            if current_stage_model.use_checkpoint_neighbor and self.training:
                neighbor_cond = checkpoint(
                    current_stage_model._neighbor_embedder_forward, 
                    neighbor_patches, 
                    use_reentrant=False
                )
            else:
                neighbor_cond = current_stage_model.neighbor_embedder(neighbor_patches)
            
            # [batch, num_h, num_w, 3, ps, ps] → [batch*num_h*num_w, 3, ps, ps]
            patches_flat = patches.view(batch_size * num_h * num_w, channels, ps, ps)
            num_patches = batch_size * num_h * num_w
            
            # 生成位置编码：每个 patch 的 (i, j) 坐标，归一化到 [0, 1]
            # i: [0, num_h-1], j: [0, num_w-1]
            i_coords = torch.arange(num_h, device=x.device).float() / (num_h - 1) if num_h > 1 else torch.zeros(1, device=x.device)
            j_coords = torch.arange(num_w, device=x.device).float() / (num_w - 1) if num_w > 1 else torch.zeros(1, device=x.device)
            # 生成所有 (i, j) 组合
            i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
            # [num_h, num_w] -> [num_h*num_w, 2]
            patch_positions = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=1)
            # 扩展到 batch: [batch*num_h*num_w, 2]
            patch_positions = patch_positions.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(-1, 2)
            
            # 随机采样策略：只验证一部分 patches 以减少计算量
            max_patches = current_stage_model.max_patches_per_step
            indices = None
            if max_patches is not None and num_patches > max_patches:
                # 随机采样 max_patches 个 patches
                indices = torch.randperm(num_patches, device=x.device)[:max_patches]
                patches_flat = patches_flat[indices]
                num_patches = max_patches
                # 也需要采样对应的 neighbor_cond 和位置编码
                neighbor_cond_flat = neighbor_cond.view(batch_size * num_h * num_w, 1, -1)[indices]
                patch_positions = patch_positions[indices]
            else:
                # neighbor_cond: [batch, num_h*num_w, embed_dim] → [batch*num_h*num_w, 1, embed_dim]
                neighbor_cond_flat = neighbor_cond.view(num_patches, 1, -1)
            
            t = torch.randint(
                0, self.timestep_list[stage], 
                (num_patches,),  # ! each patch has a time step (Key Innovation point)
                device=x.device
            )
            
            # y: [batch] → [batch*num_h*num_w] (如果采样了，需要重新计算)
            y_expanded = y.unsqueeze(1).expand(batch_size, num_h * num_w).contiguous().view(-1)
            
            # if sampled, only use the sampled patches
            if indices is not None:
                y_expanded = y_expanded[indices]
            
            # 直接处理采样后的 patches
            model_kwargs = {
                "y": y_expanded,
                "neighbor_cond": neighbor_cond_flat,
                "patch_pos": patch_positions  # [num_patches, 2] - 位置编码 (i, j) 归一化到 [0, 1]
            }
            loss_dict = current_stage_model.diffusion.training_losses(
                model=current_stage_model.model_wrapper,
                x_start=patches_flat,  # [num_patches, 3, patch_size, patch_size]
                t=t,
                model_kwargs=model_kwargs
            )
            loss = loss_dict["loss"].mean()
            
            total_loss += loss
            
            self.log(f"val_loss_stage_{stage}", loss, prog_bar=True)
            
            # Cascaded Noising: for next stage
            if stage > 0:
                with torch.no_grad():
                    # Noising the whole image (as the input for the next stage)
                    noise_t = torch.full(
                        (batch_size,), 
                        self.timestep_list[stage] - 1,
                        device=x.device,
                        dtype=torch.long
                    )
                    noise = torch.randn_like(x)
                    x = current_stage_model.diffusion.q_sample(x, noise_t, noise=noise)
        
        self.log("val_loss", total_loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        if self.config.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.scheduler.step_size, gamma=self.config.scheduler.gamma)
            return [optimizer], [scheduler]
        else:
            return optimizer
