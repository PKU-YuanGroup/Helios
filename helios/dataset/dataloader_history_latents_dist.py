import os
import pickle
import random
from collections import defaultdict

import torch
from einops import rearrange
from torch.utils.data import Dataset, Sampler


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_folders,
        history_sizes=[16, 2, 1],
        is_keep_x0=True,
        force_rebuild=False,
        return_all_vae_latent=False,
        return_prompt_raw=False,
        num_rollout_sections=3,
        single_res=False,
        single_height=384,
        single_width=640,
        seed=42,
    ):
        self.history_sizes = history_sizes
        self.is_keep_x0 = is_keep_x0
        self.force_rebuild = force_rebuild
        self.return_all_vae_latent = return_all_vae_latent
        self.return_prompt_raw = return_prompt_raw
        self.num_rollout_sections = num_rollout_sections
        self.single_res = single_res
        self.single_height = single_height
        self.single_width = single_width
        assert self.is_keep_x0, "is_keep_x0 need to be True now!"

        self.base_seed = seed
        self._epoch = 0

        if isinstance(feature_folders, str):
            self.feature_folders = [feature_folders]
        else:
            self.feature_folders = feature_folders

        self.samples = []
        self.buckets = defaultdict(list)

        for folder in self.feature_folders:
            cache_file = os.path.join(folder, "dataset_cache.pkl")
            self._process_folder(folder, cache_file)

    def _process_folder(self, folder, cache_file):
        if self.force_rebuild or not os.path.exists(cache_file):
            print(f"Building metadata cache for folder: {folder}")
            folder_samples, folder_buckets = self._build_folder_metadata(folder)

            print(f"Saving metadata cache for folder: {folder}")
            cached_data = {"samples": folder_samples, "buckets": folder_buckets}
            if not self.force_rebuild:
                with open(cache_file, "wb") as f:
                    pickle.dump(cached_data, f)
            print(f"Cached {len(folder_samples)} samples from {folder}\n")
        else:
            print(f"Loading cached metadata from: {folder}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            folder_samples = cached_data["samples"]
            folder_buckets = cached_data["buckets"]
            print(f"Loaded {len(folder_samples)} samples from cache: {folder}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(folder_samples)

        for bucket_key, indices in folder_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _build_folder_metadata(self, folder):
        feature_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(feature_files)} files in {folder}...")

        for i, feature_file in enumerate(feature_files):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(feature_files)} files")

            feature_path = os.path.join(folder, feature_file)

            # Parse filename
            parts = feature_file.split("_")
            uttid = "_".join(parts[:-3])
            num_frame = int(parts[-3])
            height = int(parts[-2])
            width = int(parts[-1].replace(".pt", ""))

            # keep length >= 121
            if num_frame < 121:
                continue

            # keep resolution
            allowed_resolutions = [
                (self.single_height, self.single_width),
                (self.single_height // 2, self.single_width // 2),
                (self.single_height // 4, self.single_width // 4),
            ]
            if self.single_res and (height, width) not in allowed_resolutions:
                continue

            bucket_key = (num_frame, height, width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": folder.rstrip("/"),
                "file_path": feature_path,
                "bucket_key": bucket_key,
                "num_frame": num_frame,
                "height": height,
                "width": width,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def prepare_stage1_latent(self, vae_latent, idx, base_vae_latent=None):
        source_latent = base_vae_latent if base_vae_latent is not None else vae_latent

        x0_latent = None
        if self.is_keep_x0:
            x0_latent = source_latent[0, :, :1, :, :].clone()
        total_sections = source_latent.shape[0]
        latent_window_size = source_latent.shape[2]
        history_window_size = sum(self.history_sizes)
        section_size = history_window_size + latent_window_size

        temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
        zero_padding_source = torch.zeros(
            temp_source_latent.shape[0],
            history_window_size,
            temp_source_latent.shape[2],
            temp_source_latent.shape[3],
            device=temp_source_latent.device,
            dtype=temp_source_latent.dtype,
        )
        continue_source_latent = torch.cat([zero_padding_source, temp_source_latent], dim=1)

        temp_vae_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
        zero_padding_vae = torch.zeros(
            temp_vae_latent.shape[0],
            history_window_size,
            temp_vae_latent.shape[2],
            temp_vae_latent.shape[3],
            device=temp_vae_latent.device,
            dtype=temp_vae_latent.dtype,
        )
        continue_vae_latent = torch.cat([zero_padding_vae, temp_vae_latent], dim=1)

        sample_seed = self.base_seed + self._epoch * 1000000 + idx
        choice_idx = torch.randint(
            0, total_sections, (1,), generator=torch.Generator().manual_seed(sample_seed)
        ).item()
        if choice_idx == 0 and x0_latent is not None:
            x0_latent = torch.zeros_like(x0_latent)

        clean_all_vae_latent = None
        if self.return_all_vae_latent:
            max_start_idx = total_sections - self.num_rollout_sections
            if max_start_idx < 0:
                raise ValueError(
                    f"Not enough sections: total_sections={total_sections}, num_rollout_sections={self.num_rollout_sections}"
                )
            start_section_idx = random.randint(0, max_start_idx)
            start_indice = start_section_idx * latent_window_size
            end_indice = start_indice + history_window_size + self.num_rollout_sections * latent_window_size
            clean_all_vae_latent = continue_source_latent[:, start_indice:end_indice, :, :]

        start_indice = choice_idx * latent_window_size
        end_indice = start_indice + section_size

        history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
        target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]

        return x0_latent, history_latent, target_latent, clean_all_vae_latent

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_f = self.samples[idx]["num_frame"]
        anchor_h = self.samples[idx]["height"]
        anchor_w = self.samples[idx]["width"]
        while True:
            sample_info = self.samples[idx]

            if (
                anchor_f != sample_info["num_frame"]
                or anchor_h != sample_info["height"]
                or anchor_w != sample_info["width"]
            ):
                idx = random.randint(0, len(self.samples) - 1)
                print("Try to find a same dim sample, retrying...")
                continue

            try:
                base_vae_latent = None
                if (anchor_h, anchor_w) in [
                    (self.single_height // 2, self.single_width // 2),
                    (self.single_height // 4, self.single_width // 4),
                ]:
                    base_file_path = (
                        sample_info["file_path"]
                        .replace("/mid", "")
                        .replace("/low", "")
                        .replace(
                            f"{self.single_height // 2}_{self.single_width // 2}",
                            f"{self.single_height}_{self.single_width}",
                        )
                        .replace(
                            f"{self.single_height // 4}_{self.single_width // 4}",
                            f"{self.single_height}_{self.single_width}",
                        )
                    )
                    base_vae_latent = torch.load(base_file_path, map_location="cpu", weights_only=False)["vae_latent"]

                feature_data = torch.load(sample_info["file_path"], map_location="cpu", weights_only=False)
                x0_latent, history_latent, target_latent, clean_all_vae_latent = self.prepare_stage1_latent(
                    feature_data["vae_latent"], idx, base_vae_latent
                )
                if self.return_prompt_raw:
                    prompt_raws = feature_data["prompt_raw"]
                break
            except Exception:
                idx = random.randint(0, len(self.samples) - 1)
                print(f"Error loading {sample_info['file_path']}, retrying...")
                file_name = os.path.basename(sample_info["file_path"])
                txt_name = f"{file_name}.txt"
                with open(txt_name, "w") as f:
                    f.write(sample_info["file_path"] + "\n")

        output_dict = {
            "uttid": sample_info["uttid"],
            "bucket_key": sample_info["bucket_key"],
            "dataset_name": sample_info["dataset_name"],
            "num_frame": sample_info["num_frame"],
            "height": sample_info["height"],
            "width": sample_info["width"],
            "x0_latents": x0_latent,
            "history_latents": history_latent,
            "target_latents": target_latent,
            "clean_all_latents": clean_all_vae_latent,
            "prompt_embeds": feature_data["prompt_embed"],
            "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
        }

        if self.return_prompt_raw:
            output_dict["prompt_raws"] = prompt_raws

        return output_dict


class BucketedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        seed=42,
        dataset_sampling_ratios=None,
        num_sp_groups=1,
        sp_world_size=1,
        global_rank=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()
        self.buckets = dataset.buckets
        self._epoch = 0

        # Distributed parameters
        self.num_sp_groups = num_sp_groups
        self.sp_world_size = sp_world_size
        self.global_rank = global_rank
        self.ith_sp_group = self.global_rank // self.sp_world_size

        self.dataset_sampling_ratios = (
            {key.rstrip("/"): value for key, value in dataset_sampling_ratios.items()}
            if dataset_sampling_ratios is not None
            else {}
        )
        self._prepare_dataset_buckets()

    def _prepare_dataset_buckets(self):
        self.dataset_buckets = {}

        for bucket_key, sample_indices in self.buckets.items():
            dataset_groups = {}
            for idx in sample_indices:
                dataset_name = self.dataset.samples[idx]["dataset_name"]
                if dataset_name not in dataset_groups:
                    dataset_groups[dataset_name] = []
                dataset_groups[dataset_name].append(idx)
            self.dataset_buckets[bucket_key] = dataset_groups

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _shard_indices_for_sp_group(self, indices):
        """
        Shard indices across SP groups, similar to DP_SP_BatchSampler.
        Each SP group gets a disjoint subset of the data.
        """
        if self.num_sp_groups == 1:
            return indices

        # Convert to tensor if it's a list
        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long)
        else:
            indices_tensor = indices

        # Pad indices if necessary to make it divisible by num_sp_groups
        total_size = len(indices_tensor)
        if total_size % self.num_sp_groups != 0:
            if not self.drop_last:
                padding_size = self.num_sp_groups - (total_size % self.num_sp_groups)
                indices_tensor = torch.cat([indices_tensor, indices_tensor[:padding_size]])
            else:
                # Truncate so every SP group gets the same number of indices.
                truncate_size = (total_size // self.num_sp_groups) * self.num_sp_groups
                indices_tensor = indices_tensor[:truncate_size]

        # Shard: each SP group gets every num_sp_groups-th element
        sp_group_indices = indices_tensor[self.ith_sp_group :: self.num_sp_groups]

        return sp_group_indices.tolist()

    def _apply_global_ratio_sampling(self):
        if not self.dataset_sampling_ratios:
            return

        dataset_sample_map = {}
        for bucket_key, dataset_groups in self.dataset_buckets.items():
            for dataset_name, indices in dataset_groups.items():
                if dataset_name not in dataset_sample_map:
                    dataset_sample_map[dataset_name] = {"indices": [], "buckets": []}
                dataset_sample_map[dataset_name]["indices"].extend(indices)
                dataset_sample_map[dataset_name]["buckets"].extend([bucket_key] * len(indices))

        total_samples = sum(len(info["indices"]) for info in dataset_sample_map.values())
        total_ratio = sum(self.dataset_sampling_ratios.values())

        sampled_dataset_map = {}
        for dataset_name, info in dataset_sample_map.items():
            if dataset_name in self.dataset_sampling_ratios:
                ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                target_samples = max(1, int(total_samples * ratio))

                indices = info["indices"]
                buckets = info["buckets"]

                if len(indices) >= target_samples:
                    selected = torch.randperm(len(indices), generator=self.generator)[:target_samples].tolist()
                    sampled_indices = [indices[i] for i in selected]
                    sampled_buckets = [buckets[i] for i in selected]
                else:
                    sampled_indices = []
                    sampled_buckets = []
                    remaining = target_samples

                    while remaining > 0:
                        repeat_count = min(remaining, len(indices))
                        selected = torch.randperm(len(indices), generator=self.generator)[:repeat_count].tolist()
                        sampled_indices.extend([indices[i] for i in selected])
                        sampled_buckets.extend([buckets[i] for i in selected])
                        remaining -= repeat_count

                sampled_dataset_map[dataset_name] = {"indices": sampled_indices, "buckets": sampled_buckets}
            else:
                sampled_dataset_map[dataset_name] = info

        new_dataset_buckets = {}
        for bucket_key in self.dataset_buckets.keys():
            new_dataset_buckets[bucket_key] = {}

        for dataset_name, info in sampled_dataset_map.items():
            indices = info["indices"]
            buckets = info["buckets"]

            for idx, bucket_key in zip(indices, buckets):
                if dataset_name not in new_dataset_buckets[bucket_key]:
                    new_dataset_buckets[bucket_key][dataset_name] = []
                new_dataset_buckets[bucket_key][dataset_name].append(idx)

        self.dataset_buckets = new_dataset_buckets

    def __iter__(self):
        # Use epoch-level seed for reproducibility
        epoch_seed = self.seed + self._epoch
        self.generator.manual_seed(epoch_seed)

        if self.dataset_sampling_ratios:
            self._apply_global_ratio_sampling()

        bucket_iterators = {}
        bucket_batches = {}

        for bucket_key, dataset_groups in self.dataset_buckets.items():
            balanced_indices = self._create_balanced_indices(dataset_groups)

            # Global shuffle before sharding (important for distributed consistency)
            if self.shuffle:
                perm = torch.randperm(len(balanced_indices), generator=self.generator).tolist()
                balanced_indices = [balanced_indices[i] for i in perm]

            # Shard indices for this SP group
            sp_group_indices = self._shard_indices_for_sp_group(balanced_indices)

            batches = []
            for i in range(0, len(sp_group_indices), self.batch_size):
                batch = sp_group_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

            if batches:
                bucket_batches[bucket_key] = batches
                bucket_iterators[bucket_key] = iter(batches)

        remaining_buckets = list(bucket_iterators.keys())

        while remaining_buckets:
            idx = torch.randint(len(remaining_buckets), (1,), generator=self.generator).item()
            bucket_key = remaining_buckets[idx]
            bucket_iter = bucket_iterators[bucket_key]

            try:
                batch = next(bucket_iter)
                yield batch
            except StopIteration:
                remaining_buckets.remove(bucket_key)

    def _create_balanced_indices(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def _equal_sampling(self, dataset_groups):
        all_indices = []
        dataset_names = list(dataset_groups.keys())

        if len(dataset_names) <= 1:
            return sum(dataset_groups.values(), [])

        min_samples = min(len(indices) for indices in dataset_groups.values())

        for dataset_name, indices in dataset_groups.items():
            if len(indices) > min_samples:
                selected = torch.randperm(len(indices), generator=self.generator)[:min_samples].tolist()
                sampled_indices = [indices[i] for i in selected]
            else:
                sampled_indices = indices
            all_indices.extend(sampled_indices)

        return all_indices

    def _ratio_sampling(self, dataset_groups):
        return sum(dataset_groups.values(), [])

    def __len__(self):
        if self.dataset_sampling_ratios:
            temp_generator = torch.Generator()
            temp_generator.manual_seed(self.seed)

            dataset_sample_map = {}
            for bucket_key, dataset_groups in self.dataset_buckets.items():
                for dataset_name, indices in dataset_groups.items():
                    if dataset_name not in dataset_sample_map:
                        dataset_sample_map[dataset_name] = []
                    dataset_sample_map[dataset_name].extend(indices)

            total_samples = sum(len(indices) for indices in dataset_sample_map.values())
            total_ratio = sum(self.dataset_sampling_ratios.values())

            sampled_total = 0
            for dataset_name, indices in dataset_sample_map.items():
                if dataset_name in self.dataset_sampling_ratios:
                    ratio = self.dataset_sampling_ratios[dataset_name] / total_ratio
                    target_samples = max(1, int(total_samples * ratio))
                    sampled_total += target_samples
                else:
                    sampled_total += len(indices)

            # Account for SP group sharding
            sp_group_samples = sampled_total // self.num_sp_groups
            if not self.drop_last and sampled_total % self.num_sp_groups != 0:
                sp_group_samples += 1

            total_batches = sp_group_samples // self.batch_size
            if not self.drop_last and sp_group_samples % self.batch_size != 0:
                total_batches += 1
            return total_batches
        else:
            total_batches = 0
            for bucket_key, dataset_groups in self.dataset_buckets.items():
                balanced_indices = self._create_balanced_indices(dataset_groups)

                # Account for SP group sharding
                sp_group_size = len(balanced_indices) // self.num_sp_groups
                if not self.drop_last and len(balanced_indices) % self.num_sp_groups != 0:
                    sp_group_size += 1

                num_batches = sp_group_size // self.batch_size
                if not self.drop_last and sp_group_size % self.batch_size != 0:
                    num_batches += 1
                total_batches += num_batches
            return total_batches


def collate_fn(batch):
    return {
        key: torch.stack([d[key] for d in batch])
        if isinstance(batch[0][key], torch.Tensor)
        else [d[key] for d in batch]
        for key in batch[0]
    }


if __name__ == "__main__":
    import torch.distributed.checkpoint as dcp
    from accelerate import Accelerator
    from torchdata.stateful_dataloader import StatefulDataLoader

    feature_folder = [
        "demo_data/ultravideo-long",
    ]
    dataloader_num_workers = 0
    batch_size = 2
    num_train_epochs = 2
    seed = 0
    output_dir = "accelerate_checkpoints"
    checkpoint_dirs = (
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if os.path.exists(output_dir)
        else []
    )

    dataset_ratios = {}
    # dataset_ratios = {
    #     "demo_data/ultravideo-long": 0.9,
    # }

    accelerator = Accelerator()
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        feature_folder,
        force_rebuild=True,
        return_all_vae_latent=True,
        return_prompt_raw=True,
        single_res=True,
        single_height=384,
        single_width=640,
        seed=seed,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        dataset_sampling_ratios=dataset_ratios,
        seed=seed,
        # num_sp_groups=get_world_size() // get_sp_world_size(),
        # sp_world_size=get_sp_world_size(),
        # global_rank=get_world_rank(),
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    dataloader = StatefulDataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=dataloader_num_workers
    )

    print(len(dataset), len(dataloader))
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")

    step = 0
    global_step = 0
    first_epoch = 0
    num_update_steps_per_epoch = len(dataloader)
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(f"Found checkpoint: {checkpoint_path}")

        accelerator.load_state(checkpoint_path)
        global_step = int(latest_checkpoint.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

        states = {
            "dataloader": dataloader,
        }
        dcp_dir = os.path.join(checkpoint_path, "distributed_checkpoint")
        dcp.load(states, checkpoint_id=dcp_dir)

        print(f"Resuming from step {global_step}, epoch {first_epoch}")

    print("Testing dataloader...")
    step = global_step
    dataset_counts = defaultdict(int)
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        for i, batch in enumerate(dataloader):
            # Get metadata
            uttid = batch["uttid"]
            num_frame = batch["num_frame"]
            height = batch["height"]
            width = batch["width"]
            bucket_key = batch["bucket_key"]

            # Get feature
            x0_latents = batch["x0_latents"]
            history_latents = batch["history_latents"]
            target_latents = batch["target_latents"]
            prompt_embeds = batch["prompt_embeds"]

            if accelerator.process_index == 0:
                # print info
                print(f" Step {step}:")
                print(f"  Batch {i}:")
                # print(f"  Data Name: {batch['dataset_name']}")
                print(f"  Batch size: {len(uttid)}")
                print(f"  Uttids: {uttid}")
                print(f"  Dimensions - frames: {num_frame[0]}, height: {height[0]}, width: {width[0]}")
                print(f"  Bucket key: {bucket_key[0]}")
                print(f"  X0 latent shape: {x0_latents.shape}")
                print(f"  History latent shape: {history_latents.shape}")
                print(f"  Context latent shape: {target_latents.shape}")
                print(f"  Prompt embed shape: {prompt_embeds.shape}")
                # print(f"  Prompt attention mask shape: {prompt_attention_masks.shape}")

                # verify
                assert all(nf == num_frame[0] for nf in num_frame), "Frame numbers not consistent in batch"
                assert all(h == height[0] for h in height), "Heights not consistent in batch"
                assert all(w == width[0] for w in width), "Widths not consistent in batch"

                print("  ✓ Batch dimensions are consistent")

            for dataset_name in batch["dataset_name"]:
                dataset_counts[dataset_name] += 1

            step += 1

            # if step == 20:
            #     checkpoint_dir = f"checkpoint-{step}"
            #     save_path = os.path.join(output_dir, checkpoint_dir)
            #     os.makedirs(save_path, exist_ok=True)

            #     if accelerator.is_main_process:
            #         print(f"Saving checkpoint at step {step}")

            #         accelerator.save_state(save_path)

            #     print(accelerator.process_index, accelerator.num_processes)
            #     states = {
            #         "dataloader": dataloader,
            #     }
            #     dcp_dir = os.path.join(save_path, "distributed_checkpoint")
            #     dcp.save(states, checkpoint_id=dcp_dir)

    print("实际采样统计:", dict(dataset_counts))
