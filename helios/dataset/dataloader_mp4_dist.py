import json
import os
import pickle
import random
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, Sampler
from video_reader import PyVideoReader

from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video


resolution_bucket_options = {
    640: [
        (768, 320),
        (768, 384),
        (640, 384),
        (768, 512),
        (576, 448),
        (512, 512),
        (448, 576),
        (512, 768),
        (384, 640),
        (384, 768),
        (320, 768),
    ],
}

length_bucket_options = {
    1: [495, 462, 429, 396, 363, 330, 297, 264, 231, 198, 165, 132, 99, 66, 33],
    2: [165, 132, 99, 66, 33],
}


def find_nearest_resolution_bucket(h, w, resolution=640):
    min_metric = float("inf")
    best_bucket = None
    for bucket_h, bucket_w in resolution_bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket


def find_nearest_length_bucket(length, stride=1):
    buckets = length_bucket_options[stride]
    min_bucket = min(buckets)
    if length < min_bucket:
        return length
    valid_buckets = [bucket for bucket in buckets if bucket <= length]
    return max(valid_buckets)


def read_cut_crop_and_resize(
    video_path, f_prime, h_prime, w_prime, stride=1, start_frame=None, end_frame=None, crop=None
):
    frame_indices = list(range(start_frame, end_frame, stride))
    assert len(frame_indices) == f_prime

    vr = PyVideoReader(video_path, threads=0)  # 0 means auto (let ffmpeg pick the optimal number)
    frames = torch.from_numpy(vr.get_batch(frame_indices)).float()

    frames = (frames / 127.5) - 1
    video = frames.permute(0, 3, 1, 2)

    s_x, e_x, s_y, e_y = crop
    video = video[:, :, s_y:e_y, s_x:e_x]

    frames, channels, h, w = video.shape
    aspect_ratio_original = h / w
    aspect_ratio_target = h_prime / w_prime

    if aspect_ratio_original >= aspect_ratio_target:
        new_h = int(w * aspect_ratio_target)
        top = (h - new_h) // 2
        bottom = top + new_h
        left = 0
        right = w
    else:
        new_w = int(h / aspect_ratio_target)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h

    # Crop the video
    cropped_video = video[:, :, top:bottom, left:right]
    # Resize the cropped video
    resized_video = torchvision.transforms.functional.resize(cropped_video, (h_prime, w_prime))
    return resized_video


def save_frames(frame_raw, fps=24, video_path="1.mp4"):
    save_list = []
    for frame in frame_raw:
        frame = (frame + 1) / 2 * 255
        frame = torchvision.transforms.transforms.ToPILImage()(frame.to(torch.uint8)).convert("RGB")
        save_list.append(frame)
        frame = None
        del frame
    export_to_video(save_list, video_path, fps=fps)

    save_list = None
    del save_list
    free_memory()


class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        json_files,
        video_folders,
        stride=1,
        base_fps=None,
        resolution=640,
        force_rebuild=True,
        single_res=False,
        single_length=False,
        single_num_frame=81,
        single_height=384,
        single_width=640,
        multi_res=False,
        id_token: Optional[str] = None,
    ):
        self.stride = stride
        self.base_fps = base_fps
        self.resolution = resolution
        self.force_rebuild = force_rebuild
        self.single_res = single_res
        self.single_height = single_height
        self.single_width = single_width
        self.single_length = single_length
        self.single_num_frame = single_num_frame
        self.multi_res = multi_res
        self.id_token = id_token or ""
        self._epoch = 0

        if isinstance(json_files, str):
            self.json_files = [json_files]
        else:
            self.json_files = json_files

        if isinstance(video_folders, str):
            self.video_folders = [video_folders]
        else:
            self.video_folders = video_folders

        assert len(self.json_files) == len(self.video_folders), (
            f"json_files ({len(self.json_files)}) and video_folders ({len(self.video_folders)}) must have the same length"
        )

        self.samples = []
        self.buckets = defaultdict(list)

        for json_file, video_folder in zip(self.json_files, self.video_folders):
            cache_file = json_file.replace(".json", "_cache.pkl").replace(".csv", "_cache.pkl")
            self._process_json_file(json_file, video_folder, cache_file)

    def _process_json_file(self, json_file, video_folder, cache_file):
        if self.force_rebuild or not os.path.exists(cache_file):
            if os.path.exists(cache_file):
                print(f"Remove {cache_file}")
                os.remove(cache_file)
            print(f"Building metadata cache for file: {json_file}")
            print(f"  Video folder: {video_folder}")
            file_samples, file_buckets = self._build_file_metadata(json_file, video_folder)

            print(f"Saving metadata cache to: {cache_file}")
            cached_data = {"samples": file_samples, "buckets": file_buckets}
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"Cached {len(file_samples)} samples from {json_file}\n")
        else:
            print(f"Loading cached metadata from: {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            file_samples = cached_data["samples"]
            file_buckets = cached_data["buckets"]
            print(f"Loaded {len(file_samples)} samples from cache: {cache_file}\n")

        sample_idx_offset = len(self.samples)
        self.samples.extend(file_samples)

        for bucket_key, indices in file_buckets.items():
            adjusted_indices = [idx + sample_idx_offset for idx in indices]
            self.buckets[bucket_key].extend(adjusted_indices)

    def _build_file_metadata(self, json_file, video_folder):
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"Scanning video folder: {video_folder}")
        existing_videos = set()
        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file.endswith(".mp4"):
                    rel_path = os.path.relpath(os.path.join(root, file), video_folder)
                    existing_videos.add(rel_path)
        print(f"Found {len(existing_videos)} video files")

        df = pd.DataFrame(
            [
                {
                    "cut": item["cut"],
                    "crop": item["crop"],
                    "path": item["path"],
                    "num_frames": item["num_frames"],
                    "width": item["resolution"]["width"],
                    "height": item["resolution"]["height"],
                    "fps": item["fps"],
                    "cap": item["cap"],
                }
                for item in data
            ]
        )

        samples = []
        buckets = defaultdict(list)
        sample_idx = 0

        print(f"Processing {len(df)} records from {json_file} with stride={self.stride}...")
        for i, row in df.iterrows():
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(df)} records")

            video_file = (
                row["path"]
                .replace("videos_clip_v1_20241111/", "")
                .replace("videos_clip_v2_20241111/", "")
                .replace("videos_clip_v4_20241111/", "")
            )
            if video_file not in existing_videos:
                print("bad video!")
                continue
            video_path = os.path.join(video_folder, video_file)

            cut_start_frame = row["cut"][0]
            cut_end_frame = row["cut"][1]
            num_frame = cut_end_frame - cut_start_frame

            if self.single_length:
                if num_frame < self.single_num_frame:
                    continue
            else:
                if num_frame < 121:
                    continue

            uttid = os.path.basename(video_file).replace(".mp4", "") + f"_{cut_start_frame}-{cut_end_frame}"
            fps = row["fps"]

            crop = row["crop"]
            width = crop[1] - crop[0]
            height = crop[3] - crop[2]

            prompt = row["cap"][0]

            # TODO need to be checked
            effective_num_frame = (num_frame + self.stride - 1) // self.stride
            bucket_num_frame = find_nearest_length_bucket(effective_num_frame, stride=self.stride)
            bucket_height, bucket_width = find_nearest_resolution_bucket(height, width, resolution=self.resolution)

            if self.single_res or self.multi_res:
                allowed_resolutions = [(self.single_height, self.single_width)]
                if self.multi_res:
                    allowed_resolutions.extend(
                        [
                            (self.single_height // 2, self.single_width // 2),
                            (self.single_height // 4, self.single_width // 4),
                        ]
                    )
                if (bucket_height, bucket_width) not in allowed_resolutions:
                    print("continue res")
                    continue
                bucket_height, bucket_width = random.choice(allowed_resolutions)

            if self.single_length:
                bucket_num_frame = self.single_num_frame

            if self.base_fps is not None:
                stride = max(int(fps / self.base_fps), 1)
                required_frames = bucket_num_frame * stride
                if required_frames >= num_frame:
                    print("continue frame")
                    continue
            else:
                stride = self.stride

            bucket_key = (bucket_num_frame, bucket_height, bucket_width)

            sample_info = {
                "uttid": uttid,
                "dataset_name": json_file.rstrip("/"),
                "video_folder": video_folder,
                "video_path": video_path,
                "bucket_key": bucket_key,
                "prompt": self.id_token + prompt,
                "fps": fps,
                "stride": stride,
                "effective_num_frame": effective_num_frame,
                "num_frame": num_frame,
                "height": height,
                "width": width,
                "bucket_num_frame": bucket_num_frame,
                "bucket_height": bucket_height,
                "bucket_width": bucket_width,
                "cut_start_frame": cut_start_frame,
                "cut_end_frame": cut_end_frame,
                "crop": crop,
            }

            samples.append(sample_info)
            buckets[bucket_key].append(sample_idx)
            sample_idx += 1

        return samples, buckets

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_h = self.samples[idx]["bucket_height"]
        anchor_w = self.samples[idx]["bucket_width"]
        anchor_f = self.samples[idx]["bucket_num_frame"]

        max_retries = 1000
        retry_count = 0

        while retry_count < max_retries:
            sample_info = self.samples[idx]

            if (
                anchor_h != sample_info["bucket_height"]
                or anchor_w != sample_info["bucket_width"]
                or anchor_f != sample_info["bucket_num_frame"]
            ):
                idx = random.randint(0, len(self.samples) - 1)
                retry_count += 1
                continue

            try:
                stride = sample_info["stride"]
                cut_start_frame = sample_info["cut_start_frame"]
                cut_end_frame = sample_info["cut_end_frame"]
                bucket_num_frame = sample_info["bucket_num_frame"]

                max_start_frame = cut_end_frame - bucket_num_frame * stride
                if max_start_frame < cut_start_frame:
                    start_frame = cut_start_frame
                else:
                    start_frame = random.randint(cut_start_frame, max_start_frame)
                end_frame = start_frame + bucket_num_frame * stride

                video_data = read_cut_crop_and_resize(
                    video_path=sample_info["video_path"],
                    f_prime=sample_info["bucket_num_frame"],
                    h_prime=sample_info["bucket_height"],
                    w_prime=sample_info["bucket_width"],
                    stride=stride,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    crop=sample_info["crop"],
                )

                return {
                    "uttid": sample_info["uttid"],
                    "bucket_key": sample_info["bucket_key"],
                    "dataset_name": sample_info["dataset_name"],
                    "video_metadata": {
                        "num_frames": sample_info["bucket_num_frame"],
                        "height": sample_info["bucket_height"],
                        "width": sample_info["bucket_width"],
                        "fps": sample_info["fps"],
                        "stride": stride,
                        "effective_num_frame": sample_info["effective_num_frame"],
                    },
                    "videos": video_data,
                    "prompts": sample_info["prompt"],
                    "first_frames_images": (video_data[0] + 1) / 2 * 255,
                }
            except Exception as e:
                print(f"Error loading {sample_info['video_path']}: {e}")
                idx = random.randint(0, len(self.samples) - 1)
                retry_count += 1

        print(f"Failed to load sample after {max_retries} retries, returning None")
        return None


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
            # If drop_last, truncate to be divisible
            if self.drop_last:
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
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    def collate_dict(data_list):
        if isinstance(data_list[0], dict):
            return {key: collate_dict([d[key] for d in data_list]) for key in data_list[0]}
        elif isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        else:
            return data_list

    return {key: collate_dict([d[key] for d in batch]) for key in batch[0]}


if __name__ == "__main__":
    import torch.distributed.checkpoint as dcp
    from accelerate import Accelerator
    from torchdata.stateful_dataloader import StatefulDataLoader

    json_file = [
        "opensoraplan/jsons/video_mixkit_513f_1997.json",
    ]
    video_folder = [
        "opensoraplan/videos",
    ]
    stride = 1
    batch_size = 2
    num_train_epochs = 1
    seed = 0
    num_workers = 8
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
    #     "/mnt/hdfs/data/ysh_new/userful_things_wan/open-sora-plan-istock/istock_v4/latents": 0.9,
    #     "/mnt/hdfs/data/ysh_new/userful_things_wan/sekai/sekai-real-walking-hq-193/latents_stride1": 0.1
    # }

    accelerator = Accelerator()
    print(accelerator.process_index, accelerator.num_processes)

    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=640,
        single_res=True,
        single_height=384,
        single_width=640,
        single_length=True,
        single_num_frame=81,
        multi_res=True,
    )
    sampler = BucketedSampler(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        dataset_sampling_ratios=dataset_ratios,
        seed=seed,
        # num_sp_groups=get_world_size() // get_sp_world_size(),
        # sp_world_size=get_sp_world_size(),
        # global_rank=get_world_rank(),
        num_sp_groups=accelerator.num_processes // 1,
        sp_world_size=1,
        global_rank=accelerator.process_index,
    )
    dataloader = StatefulDataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)

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
            bucket_key = batch["bucket_key"]
            num_frame = batch["video_metadata"]["num_frames"]
            height = batch["video_metadata"]["height"]
            width = batch["video_metadata"]["width"]

            # Get feature
            video_data = batch["videos"]
            prompt = batch["prompts"]
            first_frames_images = batch["first_frames_images"]
            first_frames_images = [torchvision.transforms.ToPILImage()(x.to(torch.uint8)) for x in first_frames_images]

            # save_frames(video_data[0].squeeze(0), video_path="1.mp4")
            # import pdb;pdb.set_trace()

            if accelerator.process_index == 0:
                # print info
                print(f" Step {step}:")
                print(f"  Batch {i}:")
                # print(f"  Data Name: {batch['dataset_name']}")
                print(f"  Batch size: {len(uttid)}")
                print(f"  Uttids: {uttid}")
                print(f"  Dimensions - frames: {num_frame[0]}, height: {height[0]}, width: {width[0]}")
                print(f"  Bucket key: {bucket_key[0]}")
                print(f"  Videos shape: {video_data.shape}")
                print(f"  Cpation: {prompt}")

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
