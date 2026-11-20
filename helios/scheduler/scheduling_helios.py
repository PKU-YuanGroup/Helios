import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, deprecate


@dataclass
class HeliosSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor
    model_outputs: torch.FloatTensor | None = None
    last_sample: torch.FloatTensor | None = None
    this_order: int | None = None


class HeliosScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,  # Following Stable diffusion 3,
        stages: int = 3,
        stage_range: List = [0, 1 / 3, 2 / 3, 1],
        gamma: float = 1 / 3,
        # For UniPC
        thresholding: bool = False,
        prediction_type: str = "flow_prediction",
        solver_order: int = 2,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        use_flow_sigmas: bool = True,
        version: str = "v1",
    ):
        self.version = version
        self.timestep_ratios = {}  # The timestep ratio for each stage
        self.timesteps_per_stage = {}  # The detailed timesteps per stage (fix max and min per stage)
        self.sigmas_per_stage = {}  # always uniform [1000, 0]
        self.start_sigmas = {}  # for start point / upsample renoise
        self.end_sigmas = {}  # for end point
        self.ori_start_sigmas = {}

        # self.init_sigmas()
        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.gamma = gamma

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def init_sigmas(self):
        """
        initialize the global timesteps and sigmas
        """
        num_train_timesteps = self.config.num_train_timesteps
        shift = self.config.shift

        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
        sigmas = torch.from_numpy(sigmas)
        timesteps = (sigmas * num_train_timesteps).clone()

        self._step_index = None
        self._begin_index = None
        self.timesteps = timesteps
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def init_sigmas_for_each_stage(self):
        """
        Init the timesteps for each stage
        """
        self.init_sigmas()

        stage_distance = []
        stages = self.config.stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        # Init the start and end point of each stage
        for i_s in range(stages):
            # To decide the start and ends point
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s + 1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.config.gamma
                corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                # corrected_sigma = 1 / (2 - ori_sigma) * ori_sigma
                start_sigma = 1 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

            if self.version == "v2":
                new_start_indice = (
                    len(self.sigmas) - torch.searchsorted(self.sigmas.flip(0), start_sigma, right=True)
                ).item()
                self.sigmas_per_stage[i_s] = self.sigmas[new_start_indice:end_indice]
                self.timesteps_per_stage[i_s] = self.timesteps[new_start_indice:end_indice]

        if self.version == "v2":
            return

        # Determine the ratio of each stage according to flow length
        tot_distance = sum(stage_distance)
        for i_s in range(stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == stages - 1:
                end_ratio = 0.9999999999999999
            else:
                end_ratio = sum(stage_distance[: i_s + 1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        # Determine the timesteps and sigmas for each stage
        for i_s in range(stages):
            timestep_ratio = self.timestep_ratios[i_s]
            # timestep_max = self.timesteps[int(timestep_ratio[0] * training_steps)]
            timestep_max = min(self.timesteps[int(timestep_ratio[0] * training_steps)], 999)
            timestep_min = self.timesteps[min(int(timestep_ratio[1] * training_steps), training_steps - 1)]
            timesteps = np.linspace(timestep_max, timestep_min, training_steps + 1)
            self.timesteps_per_stage[i_s] = (
                timesteps[:-1] if isinstance(timesteps, torch.Tensor) else torch.from_numpy(timesteps[:-1])
            )
            stage_sigmas = np.linspace(0.999, 0, training_steps + 1)
            self.sigmas_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Setting the timesteps and sigmas for each stage
        """
        self.num_inference_steps = num_inference_steps
        self.init_sigmas()

        if self.version == "v1":
            stage_timesteps = self.timesteps_per_stage[stage_index]
            timestep_max = stage_timesteps[0].item()
            timestep_min = stage_timesteps[-1].item()

            timesteps = np.linspace(
                timestep_max,
                timestep_min,
                num_inference_steps,
            )
            self.timesteps = torch.from_numpy(timesteps).to(device=device)

            stage_sigmas = self.sigmas_per_stage[stage_index]
            sigma_max = stage_sigmas[0].item()
            sigma_min = stage_sigmas[-1].item()

            ratios = np.linspace(sigma_max, sigma_min, num_inference_steps)
            sigmas = torch.from_numpy(ratios).to(device=device)
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        else:
            total_steps = len(self.timesteps_per_stage[stage_index])
            indices = np.linspace(0, total_steps - 1, num_inference_steps, dtype=int)

            self.timesteps = self.timesteps_per_stage[stage_index][indices].to(device=device)

            if stage_index == (self.config.stages - 1):
                sigmas = self.sigmas_per_stage[stage_index][indices].to(device=device)
                self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            else:
                sigmas = self.sigmas_per_stage[stage_index][indices].to(device=device)
                self.sigmas = torch.cat(
                    [sigmas, torch.tensor([self.ori_start_sigmas[stage_index + 1]], device=sigmas.device)]
                )

        self._step_index = None
        self.reset_scheduler_history()

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor] = None,
        sample: torch.FloatTensor = None,
        generator: Optional[torch.Generator] = None,
        sigma: Optional[torch.FloatTensor] = None,
        sigma_next: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[HeliosSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        assert (sigma is None) == (sigma_next is None), "sigma and sigma_next must both be None or both be not None"

        if sigma is None and sigma_next is None:
            if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
            ):
                raise ValueError(
                    (
                        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                        " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                        " one of the `scheduler.timesteps` as a timestep."
                    ),
                )

        if self.step_index is None:
            self._step_index = 0

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if sigma is None and sigma_next is None:
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return HeliosSchedulerOutput(prev_sample=prev_sample)

    # ---------------------------------- UniPC ----------------------------------
    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        if self.config.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = torch.clamp(sigma, min=1e-8)
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        sigma: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        flag = False
        if sigma is None:
            flag = True
            sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "epsilon":
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.config.prediction_type == "flow_prediction":
                if flag:
                    sigma_t = self.sigmas[self.step_index]
                else:
                    sigma_t = sigma
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, "
                    "`v_prediction`, or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        order: int = None,
        sigma: torch.Tensor = None,
        sigma_next: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        if sigma_next is None and sigma is None:
            sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        else:
            sigma_t, sigma_s0 = sigma_next, sigma
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,
        sigma_before: torch.Tensor = None,
        sigma: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            this_timestep (`int`):
                The current timestep `t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The `p` of UniC-p at this step. The effective order of accuracy should be `order + 1`.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as a required keyword argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as a required keyword argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        if sigma_before is None and sigma is None:
            sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        else:
            sigma_t, sigma_s0 = sigma, sigma_before
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def step_unipc(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor] = None,
        sample: torch.Tensor = None,
        return_dict: bool = True,
        model_outputs: list = None,
        timestep_list: list = None,
        sigma_before: torch.Tensor = None,
        sigma: torch.Tensor = None,
        sigma_next: torch.Tensor = None,
        cus_step_index: int = None,
        cus_lower_order_num: int = None,
        cus_this_order: int = None,
        cus_last_sample: torch.Tensor = None,
    ) -> Union[HeliosSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep UniPC.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        # don't change
        # print(len(self.model_outputs), len(self.timestep_list), self.disable_corrector, self.solver_p, self._begin_index)

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if cus_step_index is None:
            if self.step_index is None:
                self._step_index = 0
        else:
            self._step_index = cus_step_index

        if cus_lower_order_num is not None:
            self.lower_order_nums = cus_lower_order_num

        if cus_this_order is not None:
            self.this_order = cus_this_order

        if cus_last_sample is not None:
            self.last_sample = cus_last_sample

        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        # Convert model output using the proper conversion method
        model_output_convert = self.convert_model_output(model_output, sample=sample, sigma=sigma)

        if model_outputs is not None and timestep_list is not None:
            self.model_outputs = model_outputs[:-1]
            self.timestep_list = timestep_list[:-1]

        # print("1", self.step_index, self.timestep_list)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                sigma_before=sigma_before,
                sigma=sigma,
            )

        if model_outputs is not None and timestep_list is not None:
            model_outputs[-1] = model_output_convert
            self.model_outputs = model_outputs[1:]
            self.timestep_list = timestep_list[1:]
        else:
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
                self.timestep_list[i] = self.timestep_list[i + 1]
            self.model_outputs[-1] = model_output_convert
            self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        # change
        # print("2", self.step_index, self.timestep_list, self.lower_order_nums, self.this_order, "\n")
        # print(self._step_index, self.lower_order_nums, use_corrector, self.this_order, self.lower_order_nums)
        # 0 1 False 1 1
        # 1 2  True 2 2
        # 2 2  True 2 2
        # 3 2  True 2 2
        # 4 2  True 2 2
        # 5 2  True 2 2
        # 6 2  True 2 2
        # 7 2  True 2 2
        # 8 2  True 2 2
        # 9 2  True 1 2

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
            sigma=sigma,
            sigma_next=sigma_next,
        )

        if cus_lower_order_num is None:
            if self.lower_order_nums < self.config.solver_order:
                self.lower_order_nums += 1

        # upon completion increase step index by one
        if cus_step_index is None:
            self._step_index += 1

        if not return_dict:
            return (prev_sample, model_outputs, self.last_sample, self.this_order)

        return HeliosSchedulerOutput(
            prev_sample=prev_sample,
            model_outputs=model_outputs,
            last_sample=self.last_sample,
            this_order=self.this_order,
        )

    def reset_scheduler_history(self):
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.disable_corrector = self.config.disable_corrector
        self.solver_p = self.config.solver_p
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

    def __len__(self):
        return self.config.num_train_timesteps


if __name__ == "__main__":
    device = "cuda"

    # ---------------------- For dynamic shifting ----------------------
    from examples.scheduling_unipc_multistep_latest import UniPCMultistepScheduler

    scheduler_official = UniPCMultistepScheduler.from_pretrained("BestWishYsh/Helios-Base", subfolder="scheduler")
    scheduler_official.set_timesteps(num_inference_steps=50)
    scheduler_official.timesteps
    scheduler_official.sigmas

    # # Official
    # from scheduling_flow_match_euler_discrete_official import FlowMatchEulerDiscreteScheduler
    # scheduler_official = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    # scheduler_official.set_timesteps(num_inference_steps=50, sigmas=None)
    # scheduler_official.timesteps
    # scheduler_official.sigmas

    # import sys
    # sys.path.append("../../")
    # from helios.utils.utils_helios_base import apply_schedule_shift

    # sigmas = apply_schedule_shift(scheduler_official.sigmas, torch.ones([2, 16, 21, 48, 80]), mu=3)
    # timesteps = sigmas[:-1] * 1000.0

    # import copy
    # from diffusers.training_utils import compute_density_for_timestep_sampling

    # def get_sigmas(timesteps, n_dim=4, device="cpu", dtype=torch.float32):
    #     sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    #     schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    #     timesteps = timesteps.to(device)
    #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    # noise_scheduler_copy = copy.deepcopy(scheduler_official)

    # # Sample noise that we'll add to the latents
    # model_input = torch.ones([2, 16, 9, 88, 68])
    # noise = torch.randn_like(model_input)
    # bsz = model_input.shape[0]

    # # Sample a random timestep for each image
    # # for weighting schemes where we sample timesteps non-uniformly
    # u = compute_density_for_timestep_sampling(
    #     weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29
    # )
    # indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    # timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    # # Add noise according to flow matching.
    # # zt = (1 - texp) * x + texp * z1
    # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

    # import sys
    # sys.path.append("../../")
    # from helios.utils.utils_helios_base import apply_schedule_shift

    # sigmas = apply_schedule_shift(sigmas, noise)  # torch.Size([2, 1, 1, 1, 1])
    # timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)
    # while timesteps.ndim > 1:
    #     timesteps = timesteps.squeeze(-1)
    # ---------------------- For dynamic shifting ----------------------

    # ---------------------- For timestep shifting ----------------------
    stages = 3
    timestep_shift = 1.0
    stage_range = [0, 1 / 3, 2 / 3, 1]
    scheduler_gamma = 1 / 3
    version = "v1"
    scheduler = HeliosScheduler(
        shift=timestep_shift, stages=stages, stage_range=stage_range, gamma=scheduler_gamma, version=version
    )
    print(
        f"The start sigmas and end sigmas of each stage is Start: {scheduler.start_sigmas}, End: {scheduler.end_sigmas}, Ori_start: {scheduler.ori_start_sigmas}"
    )

    i_s = 1
    stage2_num_inference_steps_list = [3, 3, 3]
    scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s)
    scheduler.timesteps.to(dtype=torch.float32)
    scheduler.sigmas.to(dtype=torch.float32)

    # stages = 2
    # timestep_shift = 3.0
    # stage_range = [0, 1 / 2, 1]
    # scheduler_gamma = 1 / 3
    # version = "v2"
    # scheduler = HeliosScheduler(
    #     shift=timestep_shift, stages=stages, stage_range=stage_range, gamma=scheduler_gamma, version=version
    # )
    # print(
    #     f"The start sigmas and end sigmas of each stage is Start: {scheduler.start_sigmas}, End: {scheduler.end_sigmas}, Ori_start: {scheduler.ori_start_sigmas}"
    # )

    # i_s = 1
    # stage2_num_inference_steps_list = [10, 10]
    # scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s)
    # scheduler.timesteps.to(dtype=torch.float32)
    # scheduler.sigmas.to(dtype=torch.float32)

    # scheduler.timesteps_per_stage[0]
    # scheduler.sigmas_per_stage[0]
    # shift1: (999, 743.5120) -> (743.2563, 385.9723) -> (385.6146, 1.3846)
    # shift3: (999, 957.3958) -> (957.3542, 828.9170) -> (828.7885, 3.8198)

    # timesteps_1 = np.linspace(1, 1000 - 1, 1000, dtype=np.float32)[::-1].copy()
    # timesteps_1 = torch.from_numpy(timesteps_1).to(dtype=torch.float32)
    # sigmas_1 = timesteps_1 / 1000
    # sigmas_1 = apply_schedule_shift(sigmas_1, torch.ones([2, 16, 21, 48, 80]), mu=3)
    # timesteps_2 = sigmas_1 * 1000

    # import pdb;pdb.set_trace()
    # temp_sigmas = apply_schedule_shift(scheduler.timesteps / 1000, torch.ones([2, 16, 21, 48, 80]), mu=3)
    # temp_timesteps = temp_sigmas * 1000
    # while temp_timesteps.ndim > 1:
    #     temp_timesteps = temp_timesteps.squeeze(-1)
    # temp_timesteps = temp_timesteps[:-1]

    # # very important here!
    # timesteps = temp_timesteps
    # # self.scheduler.sigmas = temp_sigmas
    # scheduler.timesteps = temp_timesteps

    # ---------------------- For timestep shifting ----------------------

    # ---------------------- For dynamic shifting ----------------------

    # ---------------------- For per step sigmas & timesteps ----------------------
    # scheduler = HeliosScheduler(shift=3.0, stages=stages, stage_range=stage_range, gamma=scheduler_gamma)
    # stage2_num_inference_steps_list = [10, 10, 10]
    # i_s = 0
    # scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s)
    # scheduler.timesteps_per_stage[0]
    # scheduler.sigmas_per_stage[0]
    # scheduler.timesteps
    # scheduler.sigmas
    # ---------------------- For per step sigmas & timesteps ----------------------

    # ---------------------- For Custom step ----------------------
    # timesteps = scheduler.timesteps
    # noise_pred = torch.randn([2, 16, 10, 48, 80], device=device)
    # latents = torch.randn([2, 16, 10, 48, 80], device=device)
    # for i, t in enumerate(timesteps):
    #     print(i, t)
    #     # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    #     latents = scheduler.step_custom_unipc(noise_pred, t, latents, return_dict=False)[0]

    # def upsample_tensor(tensor, scale_factor=2):
    #     return torch.nn.functional.interpolate(
    #         tensor, scale_factor=scale_factor, mode="trilinear", align_corners=False
    #     )

    # stage2_num_inference_steps_list = [10, 10, 10]
    # noise_pred = torch.randn([2, 16, 10, 12, 20], device=device)
    # latents = torch.randn([2, 16, 10, 12, 20], device=device)
    # for stage, num_steps in enumerate(stage2_num_inference_steps_list):
    #     print(f"stage: {stage}, num_steps: {num_steps}")
    #     if stage > 0:
    #         latents = upsample_tensor(latents, scale_factor=2)
    #         noise_pred = upsample_tensor(noise_pred, scale_factor=2)

    #     scheduler.set_timesteps(num_steps, stage)
    #     timesteps = scheduler.timesteps

    #     print(f"Timesteps for stage {stage + 1}: {timesteps}")

    #     for i, t in enumerate(timesteps):
    #         # print(i, t, latents.shape)
    #         # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    #         latents = scheduler.step_unipc(noise_pred, t, latents, return_dict=False)[0]
    # ---------------------- For Custom step ----------------------
