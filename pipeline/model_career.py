"""
Vector Equities Continuous Career MTNN — continuous entity redesign for trading utility
- Per-timestep tower encoder (same 17 families)
- Continuous time fusion (year_norm + tenure + CEO change + macro) instead of discrete season embedding
- Causal career transformer over ticker sequences
- Trading heads: fwd_ret_1M/3M/6M/12M, fwd_vol, fwd_dd, entry_score (triple barrier), turnaround_prob, distress_prob

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, y):
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(y)))) + y)


class ResidualTower(nn.Module):
    def __init__(
        self, d_in: int, d_out: int = 24, d_hidden: int = 96, n_blocks: int = 1
    ):
        super().__init__()
        d_cat = d_in * 2
        self.fc1 = nn.Linear(d_cat, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_cat, d_out) if d_cat != d_out else nn.Identity()
        self.blocks = nn.ModuleList(
            [_ResBlock(d_out, d_hidden) for _ in range(max(0, n_blocks - 1))]
        )

    def forward(self, x, m):
        h = torch.cat([x * m, m], dim=-1)
        y = self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(h)))) + self.skip(h))
        for blk in self.blocks:
            y = blk(y)
        return y


class ContinuousFusion(nn.Module):
    """Gated fusion with continuous time encoding instead of discrete season embedding"""

    def __init__(
        self,
        n_towers: int,
        d_tower: int,
        d_time: int = 8,
        d_time_emb: int = 16,
        d_emb: int = 48,
        d_hidden: int = 192,
    ):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time_emb),
            nn.GELU(),
            nn.LayerNorm(d_time_emb),
            nn.Linear(d_time_emb, d_time_emb),
        )
        self.gate = nn.Linear(d_tower, 1)
        self.attn = nn.Sequential(
            nn.Linear(d_tower, d_tower), nn.Tanh(), nn.Linear(d_tower, 1)
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_tower + d_time_emb, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_emb),
        )

    def forward(self, tower_stack, time_enc):
        # tower_stack: (B, n_towers, d_tower) or (B*L, n_towers, d_tower)
        # time_enc: (B, d_time) or (B*L, d_time) or (B, L, d_time)
        # unify to (B*, n_towers, d_tower)
        orig_shape = tower_stack.shape
        if tower_stack.dim() == 3:
            tower_stack.size(0)
            scores = self.attn(tower_stack).squeeze(-1)  # B, n_towers
            weights = torch.softmax(scores, dim=-1)
            gates = torch.sigmoid(self.gate(tower_stack).squeeze(-1))
            mixed = (tower_stack * weights.unsqueeze(-1) * gates.unsqueeze(-1)).sum(
                1
            )  # B, d_tower
            t_emb = self.time_proj(time_enc)  # B, d_time_emb
            out = self.fuse(torch.cat([mixed, t_emb], dim=-1))
            return F.normalize(out, dim=-1)
        else:
            raise ValueError(f"Unexpected tower_stack shape {orig_shape}")


class CausalCareerTransformer(nn.Module):
    def __init__(
        self,
        d_emb: int = 48,
        d_model: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_emb, d_model)
        self.pos_proj = nn.Linear(1, d_model)  # year_norm positional additive
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_proj = nn.Linear(d_model, d_emb)
        self.d_model = d_model

    def forward(self, z_seq, year_norm_seq, mask_seq=None):
        """
        z_seq: (B, L, d_emb) per-timestep embeddings (career unaware)
        year_norm_seq: (B, L, 1) normalized year (0-1) for positional signal
        mask_seq: (B, L) bool True where valid, False where padded
        Returns: c_seq (B, L, d_emb) career-contextual
        """
        _B, L, _ = z_seq.shape
        x = self.input_proj(z_seq) + self.pos_proj(year_norm_seq)
        # causal mask: prevent attending to future
        causal = torch.triu(
            torch.ones(L, L, device=z_seq.device, dtype=torch.bool), diagonal=1
        )
        # src_key_padding_mask: True where padded (should be ignored)
        key_pad = None
        if mask_seq is not None:
            key_pad = ~mask_seq  # True where invalid
        out = self.encoder(x, mask=causal, src_key_padding_mask=key_pad)
        c = self.out_proj(out)
        c = F.normalize(c, dim=-1)
        return c


class SkillTowers(nn.Module):
    def __init__(self, d_emb: int, n_skills: int, d_hidden: int = 16):
        super().__init__()
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_emb, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1)
                )
                for _ in range(n_skills)
            ]
        )

    def forward(self, emb):
        return torch.cat([t(emb) for t in self.towers], dim=-1)


class EquitiesCareerMTNN(nn.Module):
    def __init__(
        self,
        fam_dims: dict,
        d_tower: int = 24,
        d_tower_hidden: int = 96,
        d_emb: int = 48,
        d_time: int = 8,
        d_time_emb: int = 16,
        d_model: int = 96,
        n_layers: int = 4,
        n_heads: int = 4,
        n_game: int = 14,
        n_skills: int = 12,
        n_sectors: int = 11,
        n_archetypes: int = 8,
        n_tower_blocks: int = 1,
        d_skill_hidden: int = 16,
        d_head_hidden: int = 64,
        mlp_heads: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.families = sorted(fam_dims)
        self.d_emb = d_emb
        self.d_time = d_time
        self.towers = nn.ModuleDict(
            {
                fam: ResidualTower(
                    fam_dims[fam],
                    d_out=d_tower,
                    d_hidden=d_tower_hidden,
                    n_blocks=n_tower_blocks,
                )
                for fam in self.families
            }
        )
        self.fusion = ContinuousFusion(
            len(self.families),
            d_tower,
            d_time=d_time,
            d_time_emb=d_time_emb,
            d_emb=d_emb,
        )
        self.career_transformer = CausalCareerTransformer(
            d_emb=d_emb,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        def head(k):
            return (
                nn.Sequential(
                    nn.Linear(d_emb, d_head_hidden),
                    nn.GELU(),
                    nn.Linear(d_head_hidden, k),
                )
                if mlp_heads
                else nn.Linear(d_emb, k)
            )

        # existing heads
        self.archetype_head = head(n_archetypes)
        self.sector_head = head(n_sectors)
        self.profile_head = head(n_game)
        self.next_profile_head = head(n_game)
        self.valuation_head = nn.Linear(d_emb, 1)
        self.market_head = nn.Linear(d_emb, 1)
        self.vol_head = nn.Linear(d_emb, 1)
        self.health_head = nn.Linear(d_emb, 1)
        self.payout_head = nn.Linear(d_emb, 1)
        self.mgmt_head = nn.Linear(d_emb, 1)
        self.own_head = nn.Linear(d_emb, 1)
        self.skill_towers = (
            SkillTowers(d_emb, n_skills, d_hidden=d_skill_hidden) if n_skills else None
        )

        # new trading heads
        self.fwd_ret_head = head(4)  # 1M,3M,6M,12M excess
        self.fwd_vol_head = nn.Linear(d_emb, 1)
        self.fwd_dd_head = nn.Linear(d_emb, 1)
        self.entry_head = nn.Linear(
            d_emb, 1
        )  # logit for triple barrier +10% before -7% 63d
        self.turnaround_head = nn.Linear(d_emb, 1)
        self.distress_head = nn.Linear(d_emb, 1)

    def encode_timestep(self, xs, ms, time_enc):
        """
        xs: dict fam -> (B, d_in) or (B*L, d_in)
        ms: dict fam -> (B, d_in) mask
        time_enc: (B, d_time) or (B*L, d_time)
        Returns: z (B, d_emb)
        """
        parts = torch.stack(
            [self.towers[fam](xs[fam], ms[fam]) for fam in self.families], dim=1
        )  # B, n_fams, d_tower
        return self.fusion(parts, time_enc)

    def forward_sequence(
        self, xs_seq, ms_seq, time_enc_seq, year_norm_seq, mask_seq=None
    ):
        """
        xs_seq: dict fam -> (B, L, d_in)
        ms_seq: dict fam -> (B, L, d_in)
        time_enc_seq: (B, L, d_time)
        year_norm_seq: (B, L, 1)
        mask_seq: (B, L) bool valid
        Returns: c_seq (B, L, d_emb), z_seq (B, L, d_emb), outputs dict per timestep
        """
        _B, L = time_enc_seq.shape[0], time_enc_seq.shape[1]
        # encode each timestep independently then stack
        z_list = []
        for seq_pos in range(L):
            xs_l = {fam: xs_seq[fam][:, seq_pos, :] for fam in self.families}
            ms_l = {fam: ms_seq[fam][:, seq_pos, :] for fam in self.families}
            t_l = time_enc_seq[:, seq_pos, :]
            z_l = self.encode_timestep(xs_l, ms_l, t_l)  # B, d_emb
            z_list.append(z_l)
        z_seq = torch.stack(z_list, dim=1)  # B, L, d_emb
        c_seq = self.career_transformer(z_seq, year_norm_seq, mask_seq)

        # heads applied to c_seq
        Bc, Lc, Dc = c_seq.shape
        flat = c_seq.reshape(Bc * Lc, Dc)
        out = {
            "archetype": self.archetype_head(flat).reshape(Bc, Lc, -1),
            "sector": self.sector_head(flat).reshape(Bc, Lc, -1),
            "profile": self.profile_head(flat).reshape(Bc, Lc, -1),
            "next_profile": self.next_profile_head(flat).reshape(Bc, Lc, -1),
            "valuation": self.valuation_head(flat).reshape(Bc, Lc),
            "market": self.market_head(flat).reshape(Bc, Lc),
            "vol": self.vol_head(flat).reshape(Bc, Lc),
            "health": self.health_head(flat).reshape(Bc, Lc),
            "payout": self.payout_head(flat).reshape(Bc, Lc),
            "mgmt": self.mgmt_head(flat).reshape(Bc, Lc),
            "own": self.own_head(flat).reshape(Bc, Lc),
            "fwd_ret": self.fwd_ret_head(flat).reshape(Bc, Lc, 4),
            "fwd_vol": self.fwd_vol_head(flat).reshape(Bc, Lc),
            "fwd_dd": self.fwd_dd_head(flat).reshape(Bc, Lc),
            "entry": self.entry_head(flat).reshape(Bc, Lc),
            "turnaround": self.turnaround_head(flat).reshape(Bc, Lc),
            "distress": self.distress_head(flat).reshape(Bc, Lc),
        }
        if self.skill_towers is not None:
            out["skills"] = self.skill_towers(flat).reshape(Bc, Lc, -1)
        return c_seq, z_seq, out

    def forward(self, xs, ms, time_enc, year_norm=None):
        # single timestep compatibility wrapper (no career)
        z = self.encode_timestep(xs, ms, time_enc)
        # dummy career with L=1
        z_seq = z.unsqueeze(1)
        yn_seq = (
            year_norm.unsqueeze(1)
            if year_norm is not None
            else torch.zeros(z.size(0), 1, 1, device=z.device)
        )
        if yn_seq.dim() == 2:
            yn_seq = yn_seq.unsqueeze(-1)
        c_seq = self.career_transformer(z_seq, yn_seq)
        c = c_seq.squeeze(1)
        flat = c
        out = {
            "archetype": self.archetype_head(flat),
            "sector": self.sector_head(flat),
            "profile": self.profile_head(flat),
            "next_profile": self.next_profile_head(flat),
            "valuation": self.valuation_head(flat).squeeze(-1),
            "market": self.market_head(flat).squeeze(-1),
            "vol": self.vol_head(flat).squeeze(-1),
            "health": self.health_head(flat).squeeze(-1),
            "payout": self.payout_head(flat).squeeze(-1),
            "mgmt": self.mgmt_head(flat).squeeze(-1),
            "own": self.own_head(flat).squeeze(-1),
            "fwd_ret": self.fwd_ret_head(flat),
            "fwd_vol": self.fwd_vol_head(flat).squeeze(-1),
            "fwd_dd": self.fwd_dd_head(flat).squeeze(-1),
            "entry": self.entry_head(flat).squeeze(-1),
            "turnaround": self.turnaround_head(flat).squeeze(-1),
            "distress": self.distress_head(flat).squeeze(-1),
        }
        if self.skill_towers is not None:
            out["skills"] = self.skill_towers(flat)
        return c, out
