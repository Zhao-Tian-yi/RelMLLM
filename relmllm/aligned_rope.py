from __future__ import annotations


def apply_aligned_rgbt_rope_patch() -> None:
    """Make each aligned RGB-T image pair share visual spatial positions."""
    import itertools

    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    cls = modeling_qwen3_vl.Qwen3VLModel
    if getattr(cls, "_rgbt_aligned_rope_patch", False):
        return

    def get_rope_index_aligned_rgbt(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        **kwargs,
    ):
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        deltas = []
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
            token_types = mm_token_type_ids[batch_idx]
            valid_mask = attention_mask[batch_idx].bool() if attention_mask is not None else None
            if valid_mask is not None:
                current_input_ids = current_input_ids[valid_mask]
                token_types = token_types[valid_mask]

            groups = []
            for key, group in itertools.groupby(
                enumerate(token_types.tolist()), lambda item: item[1]
            ):
                group = list(group)
                groups.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            positions = []
            image_index = 0
            pair_base = None
            pair_span = 0

            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    text_pos = torch.arange(
                        text_len, device=input_ids.device
                    ).view(1, -1).expand(3, -1)
                    positions.append(text_pos + current_pos)
                    current_pos += text_len
                    continue

                grid_thw = next(grid_iters[modality_type])
                span = int(max(grid_thw[1], grid_thw[2]).item()) // spatial_merge_size
                if modality_type == 1:
                    if image_index % 2 == 0:
                        pair_base = current_pos
                        pair_span = span
                        vision_start = current_pos
                        current_pos += span
                    else:
                        vision_start = pair_base if pair_base is not None else current_pos
                        current_pos = max(
                            current_pos, int(vision_start) + max(pair_span, span)
                        )
                    image_index += 1
                else:
                    vision_start = current_pos
                    current_pos += span

                positions.append(
                    self.get_vision_position_ids(
                        vision_start,
                        grid_thw,
                        1,
                        spatial_merge_size,
                        device=input_ids.device,
                    )
                )

            if positions:
                llm_positions = torch.cat(positions, dim=1).reshape(3, -1)
            else:
                llm_positions = torch.empty(
                    (3, 0), dtype=input_ids.dtype, device=input_ids.device
                )

            target_len = int(current_input_ids.shape[0])
            if llm_positions.shape[1] > target_len:
                llm_positions = llm_positions[:, :target_len]
            elif llm_positions.shape[1] < target_len:
                pad_len = target_len - llm_positions.shape[1]
                start = int(llm_positions.max().item()) + 1 if llm_positions.numel() else 0
                pad = torch.arange(
                    start, start + pad_len, device=input_ids.device
                ).view(1, -1).expand(3, -1)
                llm_positions = torch.cat([llm_positions, pad], dim=1)

            if valid_mask is not None:
                position_ids[:, batch_idx, valid_mask] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        return position_ids, torch.tensor(deltas, device=input_ids.device).unsqueeze(1)

    cls.get_rope_index = get_rope_index_aligned_rgbt
    cls._rgbt_aligned_rope_patch = True

