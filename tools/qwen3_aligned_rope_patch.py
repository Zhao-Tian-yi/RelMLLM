#!/usr/bin/env python3
from __future__ import annotations


def apply_qwen3_vl_aligned_pair_rope_patch() -> None:
    """Patch Qwen3-VL M-RoPE for aligned RGB-T image pairs.

    Qwen3-VL treats multiple images as independent visual segments. For aligned
    RGB-T grounding, the first and second image in each sample are two modality
    observations of the same spatial grid, so their vision tokens should share
    spatial coordinates. This patch makes image pairs reuse the same visual
    start position while preserving normal text positions. It also applies a
    conservative length guard for GRPO's prompt+completion forward pass, where
    Qwen3-VL can otherwise produce a small RoPE/attention-mask length mismatch.
    """

    import itertools

    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    cls = modeling_qwen3_vl.Qwen3VLModel
    if getattr(cls, "_rgbt_aligned_rope_patch", False):
        return

    def get_rope_index_aligned_rgbt(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            valid_mask = attention_mask[batch_idx].bool() if attention_mask is not None else None
            if valid_mask is not None:
                current_input_ids = current_input_ids[valid_mask]
                input_token_type = input_token_type[valid_mask]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            image_index_in_sample = 0
            pair_base = None
            pair_span = 0

            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                    continue

                grid_thw = next(grid_iters[modality_type])
                span = int(max(grid_thw[1], grid_thw[2]).item()) // spatial_merge_size

                if modality_type == 1:
                    # RGB/TIR are paired: image 0 and image 1 share spatial RoPE,
                    # image 2 and image 3 share spatial RoPE, etc.
                    if image_index_in_sample % 2 == 0:
                        pair_base = current_pos
                        pair_span = span
                        vision_start = current_pos
                        current_pos += span
                    else:
                        vision_start = pair_base if pair_base is not None else current_pos
                        current_pos = max(current_pos, int(vision_start) + max(pair_span, span))
                    image_index_in_sample += 1
                else:
                    vision_start = current_pos
                    current_pos += span

                vision_position_ids = self.get_vision_position_ids(
                    vision_start, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)

            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            else:
                llm_positions = torch.empty((3, 0), dtype=input_ids.dtype, device=input_ids.device)

            target_len = int(current_input_ids.shape[0])
            # Safety guard for Qwen3-VL + GRPO prompt/completion forwards. The
            # mismatch is usually a small image-token accounting difference.
            if llm_positions.shape[1] > target_len:
                llm_positions = llm_positions[:, :target_len]
            elif llm_positions.shape[1] < target_len:
                pad_len = target_len - llm_positions.shape[1]
                start = int(llm_positions.max().item()) + 1 if llm_positions.numel() else 0
                pad = torch.arange(start, start + pad_len, device=input_ids.device).view(1, -1).expand(3, -1)
                llm_positions = torch.cat([llm_positions, pad], dim=1)

            if valid_mask is not None:
                position_ids[:, batch_idx, valid_mask] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    cls.get_rope_index = get_rope_index_aligned_rgbt
    cls._rgbt_aligned_rope_patch = True