#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   model.py
@Time    :   2025/01/07 11:40:33
@Author  :   Stefano Izzo
@Group   :   Modal
@Version :   1.0
@Contact :   stefano.izzo@unina.it
@License :   (C)Copyright
@Project :   SCRIPT_ST_GALLEN
@Desc    :   None
"""
# %%

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Dict, Union

# %%

def _beta_distribution(k, mode=0.5):
    # Calcola i parametri alpha e beta della distribuzione Beta
    alpha = mode * (k - 2) + 1
    beta = (1 - mode) * (k - 2) + 1
    dist = torch.distributions.Beta(alpha, beta)
    return dist

class MinMaxScalerLayer(pl.LightningModule):
    def __init__(
        self,
        boundary=None,
        scaling_range=None,
    ):
        super(MinMaxScalerLayer, self).__init__()

        self.scaling_range = scaling_range

        if boundary is not None:
            assert isinstance(boundary, list), "Boundary must be a list"
            if len(boundary) > 2:
                boundaries_min = torch.tensor(boundary[::2]).unsqueeze(0)
                boundaries_max = torch.tensor(boundary[1::2]).unsqueeze(0)
                if scaling_range is not None:
                    minumums = torch.tensor(scaling_range[::2]).unsqueeze(0)
                    maximums = torch.tensor(scaling_range[1::2]).unsqueeze(0)
                else:
                    minumums = torch.tensor(0.0)
                    maximums = torch.tensor(1.0)
            else:
                boundaries_min = torch.tensor(boundary[0])
                boundaries_max = torch.tensor(boundary[1])
                if scaling_range is not None:
                    minumums = torch.tensor(scaling_range[0])
                    maximums = torch.tensor(scaling_range[1])
                else:
                    minumums = torch.tensor(0.0)
                    maximums = torch.tensor(1.0)

            self.register_buffer(
                "scaled_min", minumums
            )  # tipo to device ma non trainable
            self.register_buffer("scaled_max", maximums)
            self.register_buffer("boundaries_min", boundaries_min)
            self.register_buffer("boundaries_max", boundaries_max)
        else:
            self.register_buffer("boundaries_min", torch.tensor(0.0))
            self.register_buffer("boundaries_max", torch.tensor(1.0))
            self.register_buffer("scaled_min", torch.tensor(0.0))
            self.register_buffer("scaled_max", torch.tensor(1.0))

    def forward(self, X):
        X_std = (X - self.boundaries_min) / (self.boundaries_max - self.boundaries_min)
        return X_std * (self.scaled_max - self.scaled_min) + self.scaled_min


class ScaledSigmoid(pl.LightningModule):
    def __init__(self, alpha=1.0):
        super(ScaledSigmoid, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * x))


# %%


class AmplitudePolaritiesModel(pl.LightningModule):
    def __init__(
        self,
        n_stations,
        xyz_boundary=None,
        boundary=None,
        scaling_range=None,
        generator=None,
    ):
        super().__init__()
        self.n_stations = n_stations
        # XYZ
        if boundary is not None:
            assert (
                xyz_boundary is None
            ), "If boundary is not None, xyz_boundary must be None"
            print("Il nome della variabile è cambiato. Iniziare ad usare xyz_boundary")
            xyz_boundary = boundary

        assert xyz_boundary is not None, "xyz_boundary must be specified"
        self.xyz_boundary = xyz_boundary

        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        # self.fc4xyz = torch.nn.Linear(32, 64)
        # self.relu4xyz = torch.nn.ReLU()
        # self.fc5xyz = torch.nn.Linear(64, 128)
        # self.relu5xyz = torch.nn.ReLU()
        # self.fc6xyz = torch.nn.Linear(128, 256)
        # self.relu6xyz = torch.nn.ReLU()
        # self.fc7xyz = torch.nn.Linear(256, 128)
        # self.relu7xyz = torch.nn.ReLU()
        # self.fc8xyz = torch.nn.Linear(128, 64)
        # self.relu8xyz = torch.nn.ReLU()
        # self.fc9xyz = torch.nn.Linear(64, 32)
        # self.relu9xyz = torch.nn.ReLU()
        self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.relu10xyz = torch.nn.ReLU()
        # self.fc11xyz = torch.nn.Linear(16, 8)
        # self.relu11xyz = torch.nn.Tanh()

        self.Aconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )
        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        self.flattenA = nn.Flatten()
        self.flattenP = nn.Flatten()
        self.flatten = nn.Flatten()
        self.fc1cp0 = torch.nn.LazyLinear(1024)
        # self.fc1cp0 = torch.nn.Linear(1280, 1024)
        # self.fc1cp0 = torch.nn.Linear(1920, 1024)
        self.relu1cp0 = torch.nn.ReLU()

        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )
        self.multihead_attention2 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )

        self.fc1cp = torch.nn.Linear(1024, 512)
        self.relu1cp = torch.nn.ReLU()
        self.fc2cp = torch.nn.Linear(512, 256)
        self.relu2cp = torch.nn.ReLU()
        self.fc3cp = torch.nn.Linear(256, 128)
        self.relu3cp = torch.nn.ReLU()
        self.fc4cp = torch.nn.Linear(128, 64)
        self.relu4cp = torch.nn.ReLU()
        self.fc5cp = torch.nn.Linear(64, 32)
        self.relu5cp = torch.nn.ReLU()
        self.fc6cp = torch.nn.Linear(32, 16)
        self.relu6cp = torch.nn.ReLU()
        self.fc7cp = torch.nn.Linear(16, 8)
        self.relu7cp = torch.nn.ReLU()
        self.fc8cp = torch.nn.Linear(8, self.output_shape)
        self.relu8cp = torch.nn.Tanh()

        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0)
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation1 = torch.nn.Tanh()
        self.fc_rake_activation2 = ScaledSigmoid(alpha=3.0)

        self.fc00A = torch.nn.LazyLinear(1024)
        # self.fc00A = torch.nn.Linear(1280, 1024)
        # self.fc00A = torch.nn.Linear(1920, 1024)
        self.relu00A = torch.nn.ReLU()
        self.fc0A = torch.nn.Linear(1024, 256)
        self.relu0A = torch.nn.ReLU()
        self.fc1A = torch.nn.Linear(256, 64)
        self.relu1A = torch.nn.ReLU()
        self.fc2A = torch.nn.Linear(64, 8)
        self.relu2A = torch.nn.ReLU()
        self.gateA = torch.nn.Linear(8, 1)
        self.gateA_activation = torch.nn.Sigmoid()

    def forward(self, batch):
        xyz = batch["XYZ"]
        Amplitudes = batch["Amplitudes"]
        polarities = batch["Polarities"].unsqueeze(1)
        # Amplitudes_normalized = (
        #     Amplitudes / Amplitudes.max(1).values.unsqueeze(1)
        # ).unsqueeze(1)
        Amplitudes_maxs = Amplitudes.max(1).values.unsqueeze(1)
        Amplitudes_normalized = torch.where(
            Amplitudes == -1, -1, Amplitudes / Amplitudes_maxs
        ).unsqueeze(1)

        presence_vector = (batch["Amplitudes"] != -1).float().unsqueeze(1)
        presence_vector = presence_vector + (
            batch["Polarities"] != 0
        ).float().unsqueeze(1)
        # Amplitudes = torch.where(Amplitudes == 0, -1., Amplitudes)
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        # x = self.relu4xyz(self.fc4xyz(x))
        # x = self.relu5xyz(self.fc5xyz(x))
        # x = self.relu6xyz(self.fc6xyz(x))
        # x = self.relu7xyz(self.fc7xyz(x))
        # x = self.relu8xyz(self.fc8xyz(x))
        # x = self.relu9xyz(self.fc9xyz(x))
        x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1)
        # x = self.relu11xyz(self.fc11xyz(x))

        Amplitudes_combined = torch.cat(
            [Amplitudes_normalized, presence_vector, staz_pos, x], dim=1
        )  # (batch_size, 5, num_stazioni)

        Polarities_combined = torch.cat(
            [polarities, presence_vector, staz_pos, x], dim=1
        )

        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        Amplitudes_combined, _ = self.multihead_attention1(
            Amplitudes_combined, Amplitudes_combined, Amplitudes_combined
        )
        Polarities_combined, _ = self.multihead_attention2(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        # Applicare convoluzioni dilatate
        A_outputs = [conv(Amplitudes_combined) for conv in self.Aconvs]
        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        # Stack per confrontare lungo la nuova dimensione (len(self.dilations))
        A_outputs = [self.flattenA(co) for co in A_outputs]
        P_outputs = [self.flattenP(co) for co in P_outputs]

        yA = torch.cat(A_outputs, dim=1)
        yP = torch.cat(P_outputs, dim=1)

        yP = self.relu1cp0(self.fc1cp0(yP))
        yP = self.relu1cp(self.fc1cp(yP))
        yP = self.relu2cp(self.fc2cp(yP))
        yP = self.relu3cp(self.fc3cp(yP))
        yP = self.relu4cp(self.fc4cp(yP))
        yP = self.relu5cp(self.fc5cp(yP))
        yP = self.relu6cp(self.fc6cp(yP))
        yP = self.relu7cp(self.fc7cp(yP))

        yA = self.relu00A(self.fc00A(yA))
        yA = self.relu0A(self.fc0A(yA))
        yA = self.relu1A(self.fc1A(yA))
        yA = self.relu2A(self.fc2A(yA))
        gate = self.gateA_activation(self.gateA(yA))
        x = yA * gate + yP  #  * (1 - gate)

        # x = torch.cat([x, y], dim=1)
        x = self.relu8cp(self.fc8cp(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake_sin = self.fc_rake_activation1(rake[..., :1])
        rake_cos = self.fc_rake_activation2(rake[..., 1:])
        # x = torch.cat([strike, dip, rake], dim=1)
        x = torch.cat([strike, dip, rake_sin, rake_cos], dim=1)

        return x

    def save_parameters_correctly(self, path, verbose=True):
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        model_copy = deepcopy(self)

        if hasattr(self, "generator"):
            self.generator = generator_backup

        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        del model_copy

    def load_parameters_correctly(self, path, device=None, verbose=True):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(
        self,
        batch: Dict[str, torch.Tensor],
        zeroing_value: Union[float, List[float]],
        switch_sign=0,
        normal_noise=0,
        beta_total_concentration=0.,
        inplace=False,
        *args,
        **kwargs,
    ):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Amplitudes" in batch, "The batch must contain the Amplitudes field"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        if not isinstance(batch["Amplitudes"], torch.Tensor) or not isinstance(
            batch["Polarities"], torch.Tensor
        ):
            Xa = torch.tensor(batch["Amplitudes"]).clone()
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xa = batch["Amplitudes"].clone()
            Xp = batch["Polarities"].clone()

        assert (
            Xa.shape == Xp.shape
        ), "Amplitudes and Polarities must have the same shape"
        assert torch.all(Xa == Xa.abs()).item(), "Amplitudes must be positive"

        device = Xa.device
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        if isinstance(zeroing_value, float) and zeroing_value > 0:
            assert zeroing_value < 1, "The zeroing_value must be in the range [0, 1)"

            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            Xa[rows, cols] = -1
            Xp[rows, cols] = 0

        elif isinstance(zeroing_value, list):
            assert (
                len(zeroing_value) == 2
            ), "The zeroing_value must be a list of two elements"

            if zeroing_value[0] > 0 and zeroing_value[1] >= zeroing_value[0]:
                assert (
                    zeroing_value[0] < 1
                ), "The zeroing_value must be in the range [0, 1)"
                assert (
                    zeroing_value[1] < 1
                ), "The zeroing_value must be in the range [0, 1)"

                Xa = torch.where(Xp == 0, -1, Xa)
                rowsP, colsP, rowsA, colsA = self.zeroing_as_same(
                    zeroing_value, Xp, Xa, beta_total_concentration, self.generator
                )

                Xp[rowsP, colsP] = 0
                Xa[rowsA, colsA] = -1

            elif zeroing_value[0] > 0:
                assert (
                    zeroing_value[0] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(zeroing_value[0], Xp, self.generator)

                Xp[rows, cols] = 0

            elif zeroing_value[1] > 0:
                assert (
                    zeroing_value[1] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(
                    zeroing_value[1], Xa, self.generator, none_value=-1
                )

                Xa[rows, cols] = -1

        elif zeroing_value == 0 or zeroing_value is None:
            pass

        else:
            raise ValueError(
                "The zeroing_value must be a float or a list of two floats"
            )

        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        if normal_noise > 0:
            Xa += torch.normal(
                torch.tensor(0.0, device=self.device),
                normal_noise * torch.where(Xa == -1, 0, Xa),
                generator=self.generator,
            )

        if inplace:
            batch["Amplitudes"] = Xa
            batch["Polarities"] = Xp
        else:
            batch_copy = deepcopy(batch)
            batch_copy["Amplitudes"] = Xa
            batch_copy["Polarities"] = Xp
            return batch_copy

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  # Converti da radianti a gradi
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # Dimensione [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        # targets è un tensore di dimensione [batch_size, 3] in gradi
        targets_rad = torch.deg2rad(targets)  # Converti da gradi a radianti
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # Dimensione [batch_size, 6]

    @staticmethod
    def zeroing(zeroing_value, X, generator=None, device=None, none_value=0):
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Z = (X != none_value).sum(dim=1)
        randomizer = torch.rand(Z.shape, device=device)
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        Z_sampler_max = Z_sampler.max().item()
        P = torch.where(X != none_value, 1, 0).float()
        P /= P.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        # Crea una maschera binaria per selezionare gli elementi
        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        cols = idxs[mask]
        return rows, cols

    @staticmethod
    def zeroing_as_same(
        zeroing_values, Xp, Xa, beta_total_concentration=0.0, generator=None, device=None
    ):
        # Azzera maggiormente le Polarità rispetto alle Ampiezze, ma dove le polarità sono 0, anche le ampiezze sono 0
        assert (
            len(zeroing_values) == 2
        ), "The zeroing_values must be a list of two elements"
        if device is None:
            assert Xp.device == Xa.device, "Xp and Xa must be on the same device"
            device = Xa.device
        elif device != Xa.device or device != Xp.device:
            Xa = Xa.to(device)
            Xp = Xp.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Zp = (Xp != 0).sum(dim=1)
        Za = (Xa != -1).sum(dim=1)
        if beta_total_concentration > 2:
            dist = _beta_distribution(beta_total_concentration, 1)
            randomizerP = dist.sample(Zp.shape).to(device) * zeroing_values[0]
            randomizerA = dist.sample(Za.shape).to(device) * zeroing_values[1]
        else:
            randomizerP = torch.rand(Zp.shape, device=device) * zeroing_values[0]
            randomizerA = torch.rand(Za.shape, device=device) * zeroing_values[1]

        Z_samplerP = torch.ceil(Zp * randomizerP).to(torch.int32)
        Z_samplerA = torch.ceil(Za * randomizerA).to(torch.int32)
        Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        Z_sampler_max = Z_samplerA.max().item()
        assert (
            Z_sampler_max < Za.max().item()
        ), "Z_sampler_max must be less than Za.max().item()"

        probs = torch.where(Xp != 0, 1, 0).float()
        probs /= probs.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(probs, Z_sampler_max, generator=generator)

        rowsP = torch.arange(Z_samplerP.size(0), device=device).repeat_interleave(
            Z_samplerP
        )
        rowsA = torch.arange(Z_samplerA.size(0), device=device).repeat_interleave(
            Z_samplerA
        )

        # Crea una maschera binaria per selezionare gli elementi
        maskP = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerP.unsqueeze(1)
        maskA = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerA.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        colsP = idxs[maskP]
        colsA = idxs[maskA]
        return rowsP, colsP, rowsA, colsA

    @staticmethod
    def zeroing_as_same_dani(zeroing_values, Xp, Xa, generator=None, device=None):
        # Azzera maggiormente le Polarità rispetto alle Ampiezze, ma dove le polarità sono 0, anche le ampiezze sono 0
        assert (
            len(zeroing_values) == 2
        ), "The zeroing_values must be a list of two elements"
        if device is None:
            assert Xp.device == Xa.device, "Xp and Xa must be on the same device"
            device = Xa.device
        elif device != Xa.device or device != Xp.device:
            Xa = Xa.to(device)
            Xp = Xp.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Zp = (Xp != 0).sum(dim=1)
        Za = (Xa != -1).sum(dim=1)
        # randomizerP = torch.rand(Zp.shape, device=device)
        # randomizerA = torch.rand(Za.shape, device=device)
        # Z_samplerP = torch.ceil(Zp * randomizerP * zeroing_values[0]).to(torch.int32)
        # Z_samplerA = torch.ceil(Za * randomizerA * zeroing_values[1]).to(torch.int32)
        # Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        # Z_sampler_max = Z_samplerA.max().item()

        # Calcola quanti azzerare tra min e max % delle attive
        min_frac, max_frac = zeroing_values
        Zp_min = torch.floor(Zp * min_frac)
        Zp_max = torch.floor(Zp * max_frac)
        Za_min = torch.floor(Za * min_frac)
        Za_max = torch.floor(Za * max_frac)

        # Interpolazione random tra min e max per ogni riga
        randP = torch.rand_like(Zp, dtype=torch.float)
        randA = torch.rand_like(Za, dtype=torch.float)

        Z_samplerP = (Zp_min + (Zp_max - Zp_min) * randP).to(torch.int32)
        Z_samplerA = (Za_min + (Za_max - Za_min) * randA).to(torch.int32)

        # Sincronizza azzeramenti: ampiezze ≥ polarità
        Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        Z_sampler_max = Z_samplerA.max().item()

        assert (
            Z_sampler_max < Za.max().item()
        ), "Z_sampler_max must be less than Za.max().item()"

        probs = torch.where(Xp != 0, 1, 0).float()
        probs /= probs.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(probs, Z_sampler_max, generator=generator)

        rowsP = torch.arange(Z_samplerP.size(0), device=device).repeat_interleave(
            Z_samplerP
        )
        rowsA = torch.arange(Z_samplerA.size(0), device=device).repeat_interleave(
            Z_samplerA
        )

        # Crea una maschera binaria per selezionare gli elementi
        maskP = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerP.unsqueeze(1)
        maskA = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerA.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        colsP = idxs[maskP]
        colsA = idxs[maskA]
        return rowsP, colsP, rowsA, colsA


# %%

# if __name__ == "__main__":
#     model = AmplitudePolaritiesModel()
#     batch = {
#         "XYZ": torch.rand(10, 3),
#         "Amplitudes": torch.rand(10, 8),
#         "Polarities": torch.randint(-1, 2, (10, 8)),
#         "staz_pos": torch.rand(10, 8, 2),
#     }
#     zeroing_value = [0.5, 0.8]
#     generator = torch.Generator()
#     Xa = batch["Amplitudes"].clone()
#     Xp = batch["Polarities"].clone()
#     device = Xa.device
#     model(batch)

# %%


class OnlyPolarityModel(pl.LightningModule):
    def __init__(
        self,
        n_stations,
        xyz_boundary=None,
        boundary=None,
        scaling_range=None,
        generator=None,
    ):
        super().__init__()
        self.n_stations = n_stations
        # XYZ
        if boundary is not None:
            assert (
                xyz_boundary is None
            ), "If boundary is not None, xyz_boundary must be None"
            print("Il nome della variabile è cambiato. Iniziare ad usare xyz_boundary")
            xyz_boundary = boundary

        assert xyz_boundary is not None, "xyz_boundary must be specified"
        self.xyz_boundary = xyz_boundary

        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.relu10xyz = torch.nn.ReLU()

        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        self.flattenP = nn.Flatten()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )
        self.fc1p = torch.nn.LazyLinear(1024)
        self.relu1p = torch.nn.ReLU()
        self.fc2p = torch.nn.Linear(1024, 512)
        self.relu2p = torch.nn.ReLU()
        self.fc3p = torch.nn.Linear(512, 256)
        self.relu3p = torch.nn.ReLU()
        self.fc4p = torch.nn.Linear(256, 128)
        self.relu4p = torch.nn.ReLU()
        self.fc5p = torch.nn.Linear(128, 64)
        self.relu5p = torch.nn.ReLU()
        self.fc6p = torch.nn.Linear(64, 32)
        self.relu6p = torch.nn.ReLU()
        self.fc7p = torch.nn.Linear(32, 16)
        self.relu7p = torch.nn.ReLU()
        self.fc8p = torch.nn.Linear(16, 8)
        self.relu8p = torch.nn.ReLU()
        self.fc9p = torch.nn.Linear(8, self.output_shape)
        self.relu9p = torch.nn.Tanh()

        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0)
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation = torch.nn.Tanh()

    def forward(self, batch):
        xyz = batch["XYZ"]
        polarities = batch["Polarities"].unsqueeze(1)
        presence_vector = (batch["Polarities"] != 0).float().unsqueeze(1)
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1)

        Polarities_combined = torch.cat(
            [polarities, presence_vector, staz_pos, x], dim=1
        )

        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        Polarities_combined, _ = self.multihead_attention(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        # Applicare convoluzioni dilatate
        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        # Stack per confrontare lungo la nuova dimensione (len(self.dilations))
        P_outputs = [self.flattenP(co) for co in P_outputs]

        yP = torch.cat(P_outputs, dim=1)

        yP = self.relu1p(self.fc1p(yP))
        yP = self.relu2p(self.fc2p(yP))
        yP = self.relu3p(self.fc3p(yP))
        yP = self.relu4p(self.fc4p(yP))
        yP = self.relu5p(self.fc5p(yP))
        yP = self.relu6p(self.fc6p(yP))
        yP = self.relu7p(self.fc7p(yP))
        yP = self.relu8p(self.fc8p(yP))

        x = yP

        x = self.relu9p(self.fc9p(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake = self.fc_rake_activation(rake)
        x = torch.cat([strike, dip, rake], dim=1)

        return x

    def save_parameters_correctly(self, path, verbose=True):
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        model_copy = deepcopy(self)

        if hasattr(self, "generator"):
            self.generator = generator_backup

        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        del model_copy

    def load_parameters_correctly(self, path, device=None, verbose=True):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(self, batch, zeroing_value, switch_sign=0, *args, **kwargs):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        if not isinstance(batch["Polarities"], torch.Tensor):
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xp = batch["Polarities"].clone()

        device = Xp.device
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        assert isinstance(zeroing_value, float), "The zeroing_value must be a float"
        if zeroing_value > 0:
            assert zeroing_value <= 1, "The zeroing_value must be in the range [0, 1]"

            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            Xp[rows, cols] = 0

        else:
            pass

        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        if isinstance(batch, dict):
            batch["Polarities"] = Xp
            return batch
        else:
            return Xp

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  # Converti da radianti a gradi
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # Dimensione [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        # targets è un tensore di dimensione [batch_size, 3] in gradi
        targets_rad = torch.deg2rad(targets)  # Converti da gradi a radianti
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # Dimensione [batch_size, 6]

    @staticmethod
    def zeroing(zeroing_value, X, generator=None, device=None):
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Z = (X != 0).sum(dim=1)
        randomizer = torch.rand(Z.shape, device=device)
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        Z_sampler_max = Z_sampler.max().item()
        P = torch.where(X != 0, 1, 0).float()
        P /= P.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        # Crea una maschera binaria per selezionare gli elementi
        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        cols = idxs[mask]
        return rows, cols


# %%


class ResidualDilatedBlock1D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResidualDilatedBlock1D, self).__init__()

        # Padding per mantenere la stessa dimensione
        padding = (kernel_size // 2) * dilation

        # Prima convoluzione dilatata
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Seconda convoluzione dilatata
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Proiezione (se il numero di canali cambia)
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x  # Salviamo l'input originale per il residuo

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Se il numero di canali cambia, usiamo una proiezione per allineare le dimensioni
        if self.projection is not None:
            identity = self.projection(identity)

        # Sommiamo il residuo e applichiamo ReLU
        out += identity
        out = F.relu(out)

        return out


# %%


class CpModel(pl.LightningModule):
    def __init__(
        self,
        boundary=None,
        scaling_range=None,
    ):
        super().__init__()
        # XYZ
        self.scaler_xyz = MinMaxScalerLayer(
            boundary=boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()

        # Cp
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )
        self.flatten = nn.Flatten()
        self.fc1cp0 = torch.nn.LazyLinear(1024)
        self.relu1cp0 = torch.nn.ReLU()

        # self.multihead_attention1 = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        # self.multihead_attention2 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        self.fc1cp = torch.nn.LazyLinear(512)
        self.relu1cp = torch.nn.ReLU()
        self.fc2cp = torch.nn.Linear(512, 256)
        self.relu2cp = torch.nn.ReLU()
        self.fc3cp = torch.nn.Linear(256, 128)
        self.relu3cp = torch.nn.ReLU()
        self.fc4cp = torch.nn.Linear(128, 64)
        self.relu4cp = torch.nn.ReLU()
        self.fc5cp = torch.nn.Linear(64, 32)
        self.relu5cp = torch.nn.ReLU()
        self.fc6cp = torch.nn.Linear(32, 16)
        self.relu6cp = torch.nn.ReLU()
        self.fc7cp = torch.nn.Linear(16, 8)
        self.relu7cp = torch.nn.Tanh()
        self.fc8cp = torch.nn.Linear(11, self.output_shape)
        self.relu8cp = torch.nn.Tanh()

        self.fc_strike = torch.nn.LazyLinear(2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.LazyLinear(2)
        self.fc_dip_activation = torch.nn.Sigmoid()
        self.fc_rake = torch.nn.LazyLinear(2)
        self.fc_rake_activation = torch.nn.Tanh()

    def forward(self, batch):
        xyz = batch["XYZ"]
        Cp = batch["Cp"]
        Cp_normalized = (Cp / Cp.abs().max(1).values.unsqueeze(1)).unsqueeze(1)
        presence_vector = (batch["Cp"] != 0).float().unsqueeze(1)
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        Cp_combined = torch.cat([Cp_normalized, presence_vector, staz_pos], dim=1)

        x = self.scaler_xyz(xyz)
        x = self.fc1xyz(x)
        x = self.relu1xyz(x)

        # Applicare convoluzioni dilatate
        conv_outputs = [conv(Cp_combined) for conv in self.convs]

        # Stack per confrontare lungo la nuova dimensione (len(self.dilations))
        stacked_outputs = [self.flatten(co) for co in conv_outputs]
        y = torch.cat(stacked_outputs, dim=1)
        y = self.relu1cp0(self.fc1cp0(y))
        y = self.relu1cp(self.fc1cp(y))
        y = self.relu2cp(self.fc2cp(y))
        y = self.relu3cp(self.fc3cp(y))
        y = self.relu4cp(self.fc4cp(y))
        y = self.relu5cp(self.fc5cp(y))
        y = self.relu6cp(self.fc6cp(y))
        y = self.relu7cp(self.fc7cp(y))
        x = torch.cat([x, y], dim=1)
        x = self.relu8cp(self.fc8cp(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake = self.fc_rake_activation(rake)
        x = torch.cat([strike, dip, rake], dim=1)

        return x

    def save_parameters_correctly(self, path):
        device = self.device
        self.cpu()
        torch.save(self.state_dict(), path)
        self.to(device)
        print(f"Model saved to {path}")

    def load_parameters_correctly(self, path, device=None):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        print(f"Model loaded from {path}")

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  # Converti da radianti a gradi
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # Dimensione [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        # targets è un tensore di dimensione [batch_size, 3] in gradi
        targets_rad = torch.deg2rad(targets)  # Converti da gradi a radianti
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # Dimensione [batch_size, 6]

    @staticmethod
    def preprocessing(batch, zeroing_value, generator=None):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Cp" in batch, "The batch must contain the Cp field"

        X = batch["Cp"].clone()

        if zeroing_value > 0:
            device = X.device
            if generator is None:
                generator = torch.Generator(device=device)
                generator.manual_seed(42)

            Z = (X != 0).sum(dim=1)
            randomizer = torch.rand(Z.shape, device=device)
            Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
            Z_sampler_max = Z_sampler.max().item()
            P = torch.where(X != 0, 1, 0).float()
            P /= P.sum(dim=1, keepdim=True)
            idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

            rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
                Z_sampler
            )

            # Crea una maschera binaria per selezionare gli elementi
            mask = torch.arange(idxs.size(1), device=device).expand(
                len(idxs), -1
            ) < Z_sampler.unsqueeze(1)

            # Usa la maschera per ottenere il risultato finale
            cols = idxs[mask]

            X[rows, cols] = 0

        if isinstance(batch, dict):
            batch["Cp"] = X
            # batch["presence"] = (X != 0).float()
            return batch
        else:
            return X  # , (X != 0).float()


# %%


class AmplitudePolaritiesEqualModel(pl.LightningModule):
    def __init__(
        self,
        n_stations,
        xyz_boundary=None,
        boundary=None,
        scaling_range=None,
        generator=None,
    ):
        super().__init__()
        self.n_stations = n_stations
        # XYZ
        if boundary is not None:
            assert (
                xyz_boundary is None
            ), "If boundary is not None, xyz_boundary must be None"
            print("Il nome della variabile è cambiato. Iniziare ad usare xyz_boundary")
            xyz_boundary = boundary

        assert xyz_boundary is not None, "xyz_boundary must be specified"
        self.xyz_boundary = xyz_boundary

        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        # self.fc4xyz = torch.nn.Linear(32, 64)
        # self.relu4xyz = torch.nn.ReLU()
        # self.fc5xyz = torch.nn.Linear(64, 128)
        # self.relu5xyz = torch.nn.ReLU()
        # self.fc6xyz = torch.nn.Linear(128, 256)
        # self.relu6xyz = torch.nn.ReLU()
        # self.fc7xyz = torch.nn.Linear(256, 128)
        # self.relu7xyz = torch.nn.ReLU()
        # self.fc8xyz = torch.nn.Linear(128, 64)
        # self.relu8xyz = torch.nn.ReLU()
        # self.fc9xyz = torch.nn.Linear(64, 32)
        # self.relu9xyz = torch.nn.ReLU()
        self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.relu10xyz = torch.nn.ReLU()
        # self.fc11xyz = torch.nn.Linear(16, 8)
        # self.relu11xyz = torch.nn.Tanh()

        self.Aconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )
        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        self.flattenA = nn.Flatten()
        self.flattenP = nn.Flatten()
        self.flatten = nn.Flatten()

        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )
        self.multihead_attention2 = nn.MultiheadAttention(
            embed_dim=5, num_heads=5, batch_first=True
        )

        self.fc0P = torch.nn.LazyLinear(1024)
        self.relu0P = torch.nn.ReLU()
        self.fc1P = torch.nn.Linear(1024, 512)
        self.relu1P = torch.nn.ReLU()
        self.fc2P = torch.nn.Linear(512, 256)
        self.relu2P = torch.nn.ReLU()
        self.fc3P = torch.nn.Linear(256, 128)
        self.relu3P = torch.nn.ReLU()
        self.fc4P = torch.nn.Linear(128, 64)
        self.relu4P = torch.nn.ReLU()
        self.fc5P = torch.nn.Linear(64, 32)
        self.relu5P = torch.nn.ReLU()
        self.fc6P = torch.nn.Linear(32, 16)
        self.relu6P = torch.nn.ReLU()
        self.fc7P = torch.nn.Linear(16, 8)
        self.relu7P = torch.nn.ReLU()

        self.fc0A = torch.nn.LazyLinear(1024)
        self.relu0A = torch.nn.ReLU()
        self.fc1A = torch.nn.Linear(1024, 512)
        self.relu1A = torch.nn.ReLU()
        self.fc2A = torch.nn.Linear(512, 256)
        self.relu2A = torch.nn.ReLU()
        self.fc3A = torch.nn.Linear(256, 128)
        self.relu3A = torch.nn.ReLU()
        self.fc4A = torch.nn.Linear(128, 64)
        self.relu4A = torch.nn.ReLU()
        self.fc5A = torch.nn.Linear(64, 32)
        self.relu5A = torch.nn.ReLU()
        self.fc6A = torch.nn.Linear(32, 16)
        self.relu6A = torch.nn.ReLU()
        self.fc7A = torch.nn.Linear(16, 8)
        self.relu7A = torch.nn.ReLU()

        self.fc_final = torch.nn.Linear(16, self.output_shape)
        self.relu_final = torch.nn.Tanh()
        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0)
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation = torch.nn.Tanh()

    def forward(self, batch):
        xyz = batch["XYZ"]
        Amplitudes = batch["Amplitudes"]
        polarities = batch["Polarities"].unsqueeze(1)
        # Amplitudes_normalized = (
        #     Amplitudes / Amplitudes.max(1).values.unsqueeze(1)
        # ).unsqueeze(1)
        Amplitudes_maxs = Amplitudes.max(1).values.unsqueeze(1)
        Amplitudes_normalized = torch.where(
            Amplitudes == -1, -1, Amplitudes / Amplitudes_maxs
        ).unsqueeze(1)

        presence_vector = (batch["Amplitudes"] != -1).float().unsqueeze(1)
        presence_vector = presence_vector + (
            batch["Polarities"] != 0
        ).float().unsqueeze(1)
        # Amplitudes = torch.where(Amplitudes == 0, -1., Amplitudes)
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        # x = self.relu4xyz(self.fc4xyz(x))
        # x = self.relu5xyz(self.fc5xyz(x))
        # x = self.relu6xyz(self.fc6xyz(x))
        # x = self.relu7xyz(self.fc7xyz(x))
        # x = self.relu8xyz(self.fc8xyz(x))
        # x = self.relu9xyz(self.fc9xyz(x))
        x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1)
        # x = self.relu11xyz(self.fc11xyz(x))

        Amplitudes_combined = torch.cat(
            [Amplitudes_normalized, presence_vector, staz_pos, x], dim=1
        )  # (batch_size, 5, num_stazioni)

        Polarities_combined = torch.cat(
            [polarities, presence_vector, staz_pos, x], dim=1
        )

        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        Amplitudes_combined, _ = self.multihead_attention1(
            Amplitudes_combined, Amplitudes_combined, Amplitudes_combined
        )
        Polarities_combined, _ = self.multihead_attention2(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        # Applicare convoluzioni dilatate
        A_outputs = [conv(Amplitudes_combined) for conv in self.Aconvs]
        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        # Stack per confrontare lungo la nuova dimensione (len(self.dilations))
        A_outputs = [self.flattenA(co) for co in A_outputs]
        P_outputs = [self.flattenP(co) for co in P_outputs]

        yA = torch.cat(A_outputs, dim=1)
        yP = torch.cat(P_outputs, dim=1)

        yP = self.relu0P(self.fc0P(yP))
        yP = self.relu1P(self.fc1P(yP))
        yP = self.relu2P(self.fc2P(yP))
        yP = self.relu3P(self.fc3P(yP))
        yP = self.relu4P(self.fc4P(yP))
        yP = self.relu5P(self.fc5P(yP))
        yP = self.relu6P(self.fc6P(yP))
        yP = self.relu7P(self.fc7P(yP))

        yA = self.relu0A(self.fc0A(yA))
        yA = self.relu1A(self.fc1A(yA))
        yA = self.relu2A(self.fc2A(yA))
        yA = self.relu3A(self.fc3A(yA))
        yA = self.relu4A(self.fc4A(yA))
        yA = self.relu5A(self.fc5A(yA))
        yA = self.relu6A(self.fc6A(yA))
        yA = self.relu7A(self.fc7A(yA))

        x = torch.cat([yA, yP], dim=1)
        x = self.relu_final(self.fc_final(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake = self.fc_rake_activation(rake)
        x = torch.cat([strike, dip, rake], dim=1)

        return x

    def save_parameters_correctly(self, path, verbose=True):
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        model_copy = deepcopy(self)

        if hasattr(self, "generator"):
            self.generator = generator_backup

        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        del model_copy

    def load_parameters_correctly(self, path, device=None, verbose=True):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(
        self,
        batch,
        zeroing_value,
        switch_sign=0,
        normal_noise=0,
        inplace=False,
        *args,
        **kwargs,
    ):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Amplitudes" in batch, "The batch must contain the Amplitudes field"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        if not isinstance(batch["Amplitudes"], torch.Tensor) or not isinstance(
            batch["Polarities"], torch.Tensor
        ):
            Xa = torch.tensor(batch["Amplitudes"]).clone()
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xa = batch["Amplitudes"].clone()
            Xp = batch["Polarities"].clone()

        assert (
            Xa.shape == Xp.shape
        ), "Amplitudes and Polarities must have the same shape"
        assert torch.all(Xa == Xa.abs()).item(), "Amplitudes must be positive"

        device = Xa.device
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        if isinstance(zeroing_value, float) and zeroing_value > 0:
            assert zeroing_value < 1, "The zeroing_value must be in the range [0, 1)"

            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            Xa[rows, cols] = -1
            Xp[rows, cols] = 0

        elif isinstance(zeroing_value, list):
            assert (
                len(zeroing_value) == 2
            ), "The zeroing_value must be a list of two elements"

            if zeroing_value[0] > 0 and zeroing_value[1] >= zeroing_value[0]:
                assert (
                    zeroing_value[0] < 1
                ), "The zeroing_value must be in the range [0, 1)"
                assert (
                    zeroing_value[1] < 1
                ), "The zeroing_value must be in the range [0, 1)"

                Xa = torch.where(Xp == 0, -1, Xa)
                rowsP, colsP, rowsA, colsA = self.zeroing_as_same(
                    zeroing_value, Xp, Xa, self.generator
                )

                Xp[rowsP, colsP] = 0
                Xa[rowsA, colsA] = -1

            elif zeroing_value[0] > 0:
                assert (
                    zeroing_value[0] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(zeroing_value[0], Xp, self.generator)

                Xp[rows, cols] = 0

            elif zeroing_value[1] > 0:
                assert (
                    zeroing_value[1] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(
                    zeroing_value[1], Xa, self.generator, none_value=-1
                )

                Xa[rows, cols] = -1

        elif zeroing_value == 0 or zeroing_value is None:
            pass

        else:
            raise ValueError(
                "The zeroing_value must be a float or a list of two floats"
            )

        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        if normal_noise > 0:
            Xa += torch.normal(
                torch.tensor(0.0, device=self.device),
                normal_noise * torch.where(Xa == -1, 0, Xa),
                generator=self.generator,
            )

        if inplace:
            batch["Amplitudes"] = Xa
            batch["Polarities"] = Xp
        else:
            batch_copy = deepcopy(batch)
            batch_copy["Amplitudes"] = Xa
            batch_copy["Polarities"] = Xp
            return batch_copy

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  # Converti da radianti a gradi
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # Dimensione [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        # targets è un tensore di dimensione [batch_size, 3] in gradi
        targets_rad = torch.deg2rad(targets)  # Converti da gradi a radianti
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # Dimensione [batch_size, 6]

    @staticmethod
    def zeroing(zeroing_value, X, generator=None, device=None, none_value=0):
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Z = (X != none_value).sum(dim=1)
        randomizer = torch.rand(Z.shape, device=device)
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        Z_sampler_max = Z_sampler.max().item()
        P = torch.where(X != none_value, 1, 0).float()
        P /= P.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        # Crea una maschera binaria per selezionare gli elementi
        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        cols = idxs[mask]
        return rows, cols

    @staticmethod
    def zeroing_as_same(zeroing_values, Xp, Xa, generator=None, device=None):
        # Azzera maggiormente le Polarità rispetto alle Ampiezze, ma dove le polarità sono 0, anche le ampiezze sono 0
        assert (
            len(zeroing_values) == 2
        ), "The zeroing_values must be a list of two elements"
        if device is None:
            assert Xp.device == Xa.device, "Xp and Xa must be on the same device"
            device = Xa.device
        elif device != Xa.device or device != Xp.device:
            Xa = Xa.to(device)
            Xp = Xp.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Zp = (Xp != 0).sum(dim=1)
        Za = (Xa != -1).sum(dim=1)
        randomizerP = torch.rand(Zp.shape, device=device)
        randomizerA = torch.rand(Za.shape, device=device)
        Z_samplerP = torch.ceil(Zp * randomizerP * zeroing_values[0]).to(torch.int32)
        Z_samplerA = torch.ceil(Za * randomizerA * zeroing_values[1]).to(torch.int32)
        Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        Z_sampler_max = Z_samplerA.max().item()
        assert (
            Z_sampler_max < Za.max().item()
        ), "Z_sampler_max must be less than Za.max().item()"

        probs = torch.where(Xp != 0, 1, 0).float()
        probs /= probs.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(probs, Z_sampler_max, generator=generator)

        rowsP = torch.arange(Z_samplerP.size(0), device=device).repeat_interleave(
            Z_samplerP
        )
        rowsA = torch.arange(Z_samplerA.size(0), device=device).repeat_interleave(
            Z_samplerA
        )

        # Crea una maschera binaria per selezionare gli elementi
        maskP = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerP.unsqueeze(1)
        maskA = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerA.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        colsP = idxs[maskP]
        colsA = idxs[maskA]
        return rowsP, colsP, rowsA, colsA


# %%
class ThreeBranchModel(pl.LightningModule):
    def __init__(
        self,
        n_stations,
        xyz_boundary=None,
        boundary=None,
        scaling_range=None,
        generator=None,
    ):
        super().__init__()
        self.n_stations = n_stations
        # XYZ
        if boundary is not None:
            assert (
                xyz_boundary is None
            ), "If boundary is not None, xyz_boundary must be None"
            print("Il nome della variabile è cambiato. Iniziare ad usare xyz_boundary")
            xyz_boundary = boundary

        assert xyz_boundary is not None, "xyz_boundary must be specified"
        self.xyz_boundary = xyz_boundary

        self.scaler_xyz = MinMaxScalerLayer(
            boundary=xyz_boundary,
            scaling_range=scaling_range,
        )
        self.scaler_xy = MinMaxScalerLayer(
            boundary=xyz_boundary[0:4],
            scaling_range=scaling_range[0:4],
        )

        self.generator = generator
        self.output_shape = 6

        self.fc1xyz = torch.nn.Linear(3, 3)
        self.relu1xyz = torch.nn.ReLU()
        self.fc2xyz = torch.nn.Linear(3, 16)
        self.relu2xyz = torch.nn.ReLU()
        self.fc3xyz = torch.nn.Linear(16, 32)
        self.relu3xyz = torch.nn.ReLU()
        # self.fc4xyz = torch.nn.Linear(32, 64)
        # self.relu4xyz = torch.nn.ReLU()
        # self.fc5xyz = torch.nn.Linear(64, 128)
        # self.relu5xyz = torch.nn.ReLU()
        # self.fc6xyz = torch.nn.Linear(128, 256)
        # self.relu6xyz = torch.nn.ReLU()
        # self.fc7xyz = torch.nn.Linear(256, 128)
        # self.relu7xyz = torch.nn.ReLU()
        # self.fc8xyz = torch.nn.Linear(128, 64)
        # self.relu8xyz = torch.nn.ReLU()
        # self.fc9xyz = torch.nn.Linear(64, 32)
        # self.relu9xyz = torch.nn.ReLU()
        # self.fc10xyz = torch.nn.Linear(32, self.n_stations)
        self.fc10xyz = torch.nn.Linear(32, 16)
        self.relu10xyz = torch.nn.ReLU()
        self.fc11xyz = torch.nn.Linear(16, 8)
        self.relu11xyz = torch.nn.ReLU()

        self.Aconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )
        self.Pconvs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, dilation=d)
                # ResidualDilatedBlock1D(in_channels=5, out_channels=32, kernel_size=3, dilation=d)
                for d in [1, 2, 3, 4]
            ]
        )

        self.flattenA = nn.Flatten()
        self.flattenP = nn.Flatten()
        self.flatten = nn.Flatten()
        self.fc1cp0 = torch.nn.LazyLinear(1024)
        # self.fc1cp0 = torch.nn.Linear(1280, 1024)
        # self.fc1cp0 = torch.nn.Linear(1920, 1024)
        self.relu1cp0 = torch.nn.ReLU()

        self.multihead_attention1 = nn.MultiheadAttention(
            embed_dim=4, num_heads=4, batch_first=True
        )
        self.multihead_attention2 = nn.MultiheadAttention(
            embed_dim=4, num_heads=4, batch_first=True
        )

        self.fc1cp = torch.nn.Linear(1024, 512)
        self.relu1cp = torch.nn.ReLU()
        self.fc2cp = torch.nn.Linear(512, 256)
        self.relu2cp = torch.nn.ReLU()
        self.fc3cp = torch.nn.Linear(256, 128)
        self.relu3cp = torch.nn.ReLU()
        self.fc4cp = torch.nn.Linear(128, 64)
        self.relu4cp = torch.nn.ReLU()
        self.fc5cp = torch.nn.Linear(64, 32)
        self.relu5cp = torch.nn.ReLU()
        self.fc6cp = torch.nn.Linear(32, 16)
        self.relu6cp = torch.nn.ReLU()
        self.fc7cp = torch.nn.Linear(16, 8)
        self.relu7cp = torch.nn.ReLU()
        self.fc8cp = torch.nn.Linear(8, self.output_shape)
        self.relu8cp = torch.nn.Tanh()

        self.fc_strike = torch.nn.Linear(self.output_shape, 2)
        self.fc_strike_activation = torch.nn.Tanh()
        self.fc_dip = torch.nn.Linear(self.output_shape, 2)
        self.fc_dip_activation = ScaledSigmoid(alpha=3.0)
        self.fc_rake = torch.nn.Linear(self.output_shape, 2)
        self.fc_rake_activation = torch.nn.Tanh()

        self.fc00A = torch.nn.LazyLinear(1024)
        # self.fc00A = torch.nn.Linear(1280, 1024)
        # self.fc00A = torch.nn.Linear(1920, 1024)
        self.relu00A = torch.nn.ReLU()
        self.fc0A = torch.nn.Linear(1024, 256)
        self.relu0A = torch.nn.ReLU()
        self.fc1A = torch.nn.Linear(256, 64)
        self.relu1A = torch.nn.ReLU()
        self.fc2A = torch.nn.Linear(64, 8)
        self.relu2A = torch.nn.ReLU()
        self.gateA = torch.nn.Linear(8, 1)
        self.gateA_activation = torch.nn.Sigmoid()

    def forward(self, batch):
        xyz = batch["XYZ"]
        Amplitudes = batch["Amplitudes"]
        polarities = batch["Polarities"].unsqueeze(1)
        # Amplitudes_normalized = (
        #     Amplitudes / Amplitudes.max(1).values.unsqueeze(1)
        # ).unsqueeze(1)
        Amplitudes_maxs = Amplitudes.max(1).values.unsqueeze(1)
        Amplitudes_normalized = torch.where(
            Amplitudes == -1, -1, Amplitudes / Amplitudes_maxs
        ).unsqueeze(1)

        presence_vector = (batch["Amplitudes"] != -1).float().unsqueeze(1)
        presence_vector = presence_vector + (
            batch["Polarities"] != 0
        ).float().unsqueeze(1)
        # Amplitudes = torch.where(Amplitudes == 0, -1., Amplitudes)
        staz_pos = batch["staz_pos"]  # (batch_size, num_stazioni, 2)
        staz_pos = self.scaler_xy(staz_pos)
        staz_pos = staz_pos.permute(0, 2, 1)

        x = self.scaler_xyz(xyz)
        x = self.relu1xyz(self.fc1xyz(x))
        x = self.relu2xyz(self.fc2xyz(x))
        x = self.relu3xyz(self.fc3xyz(x))
        # x = self.relu4xyz(self.fc4xyz(x))
        # x = self.relu5xyz(self.fc5xyz(x))
        # x = self.relu6xyz(self.fc6xyz(x))
        # x = self.relu7xyz(self.fc7xyz(x))
        # x = self.relu8xyz(self.fc8xyz(x))
        # x = self.relu9xyz(self.fc9xyz(x))
        # x = self.relu10xyz(self.fc10xyz(x)).unsqueeze(1)
        x = self.relu10xyz(self.fc10xyz(x))
        x = self.relu11xyz(self.fc11xyz(x))

        Amplitudes_combined = torch.cat(
            [Amplitudes_normalized, presence_vector, staz_pos], dim=1
        )  # (batch_size, 4, num_stazioni)

        Polarities_combined = torch.cat([polarities, presence_vector, staz_pos], dim=1)

        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)
        Amplitudes_combined, _ = self.multihead_attention1(
            Amplitudes_combined, Amplitudes_combined, Amplitudes_combined
        )
        Polarities_combined, _ = self.multihead_attention2(
            Polarities_combined, Polarities_combined, Polarities_combined
        )
        Amplitudes_combined = Amplitudes_combined.permute(0, 2, 1)
        Polarities_combined = Polarities_combined.permute(0, 2, 1)

        # Applicare convoluzioni dilatate
        A_outputs = [conv(Amplitudes_combined) for conv in self.Aconvs]
        P_outputs = [conv(Polarities_combined) for conv in self.Pconvs]

        # Stack per confrontare lungo la nuova dimensione (len(self.dilations))
        A_outputs = [self.flattenA(co) for co in A_outputs]
        P_outputs = [self.flattenP(co) for co in P_outputs]

        yA = torch.cat(A_outputs, dim=1)
        yP = torch.cat(P_outputs, dim=1)

        yP = self.relu1cp0(self.fc1cp0(yP))
        yP = self.relu1cp(self.fc1cp(yP))
        yP = self.relu2cp(self.fc2cp(yP))
        yP = self.relu3cp(self.fc3cp(yP))
        yP = self.relu4cp(self.fc4cp(yP))
        yP = self.relu5cp(self.fc5cp(yP))
        yP = self.relu6cp(self.fc6cp(yP))
        yP = self.relu7cp(self.fc7cp(yP))

        yA = self.relu00A(self.fc00A(yA))
        yA = self.relu0A(self.fc0A(yA))
        yA = self.relu1A(self.fc1A(yA))
        yA = self.relu2A(self.fc2A(yA))
        gate = self.gateA_activation(self.gateA(yA))
        x = x + yA * gate + yP  #  * (1 - gate)

        # x = torch.cat([x, y], dim=1)
        x = self.relu8cp(self.fc8cp(x))

        strike = self.fc_strike(x)
        strike = self.fc_strike_activation(strike)
        dip = self.fc_dip(x)
        dip = self.fc_dip_activation(dip)
        rake = self.fc_rake(x)
        rake = self.fc_rake_activation(rake)
        x = torch.cat([strike, dip, rake], dim=1)

        return x

    def save_parameters_correctly(self, path, verbose=True):
        if hasattr(self, "generator"):
            generator_backup = self.generator
            self.generator = None

        model_copy = deepcopy(self)

        if hasattr(self, "generator"):
            self.generator = generator_backup

        model_copy.cpu()
        torch.save(model_copy.state_dict(), path)
        if verbose:
            print(f"Model saved to {path}")
        del model_copy

    def load_parameters_correctly(self, path, device=None, verbose=True):
        if device is None:
            device = self.device
        self.cpu()
        self.load_state_dict(torch.load(path))
        self.to(device)
        if verbose:
            print(f"Model loaded from {path}")

    def preprocessing(
        self,
        batch,
        zeroing_value,
        switch_sign=0,
        normal_noise=0,
        inplace=False,
        *args,
        **kwargs,
    ):
        assert isinstance(batch, dict), "The batch must be a dictionary"
        assert "Amplitudes" in batch, "The batch must contain the Amplitudes field"
        assert "Polarities" in batch, "The batch must contain the Polarities field"

        if not isinstance(batch["Amplitudes"], torch.Tensor) or not isinstance(
            batch["Polarities"], torch.Tensor
        ):
            Xa = torch.tensor(batch["Amplitudes"]).clone()
            Xp = torch.tensor(batch["Polarities"]).clone()
        else:
            Xa = batch["Amplitudes"].clone()
            Xp = batch["Polarities"].clone()

        assert (
            Xa.shape == Xp.shape
        ), "Amplitudes and Polarities must have the same shape"
        assert torch.all(Xa == Xa.abs()).item(), "Amplitudes must be positive"

        device = Xa.device
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(42)

        if isinstance(zeroing_value, float) and zeroing_value > 0:
            assert zeroing_value < 1, "The zeroing_value must be in the range [0, 1)"

            rows, cols = self.zeroing(zeroing_value, Xp, self.generator)

            Xa[rows, cols] = -1
            Xp[rows, cols] = 0

        elif isinstance(zeroing_value, list):
            assert (
                len(zeroing_value) == 2
            ), "The zeroing_value must be a list of two elements"

            if zeroing_value[0] > 0 and zeroing_value[1] >= zeroing_value[0]:
                assert (
                    zeroing_value[0] < 1
                ), "The zeroing_value must be in the range [0, 1)"
                assert (
                    zeroing_value[1] < 1
                ), "The zeroing_value must be in the range [0, 1)"

                Xa = torch.where(Xp == 0, -1, Xa)
                rowsP, colsP, rowsA, colsA = self.zeroing_as_same(
                    zeroing_value, Xp, Xa, self.generator
                )

                Xp[rowsP, colsP] = 0
                Xa[rowsA, colsA] = -1

            elif zeroing_value[0] > 0:
                assert (
                    zeroing_value[0] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(zeroing_value[0], Xp, self.generator)

                Xp[rows, cols] = 0

            elif zeroing_value[1] > 0:
                assert (
                    zeroing_value[1] <= 1
                ), "The zeroing_value must be in the range [0, 1]"

                rows, cols = self.zeroing(
                    zeroing_value[1], Xa, self.generator, none_value=-1
                )

                Xa[rows, cols] = -1

        elif zeroing_value == 0 or zeroing_value is None:
            pass

        else:
            raise ValueError(
                "The zeroing_value must be a float or a list of two floats"
            )

        if switch_sign > 0:
            rows, cols = self.zeroing(switch_sign, Xp, self.generator)
            Xp[rows, cols] = -Xp[rows, cols]

        if normal_noise > 0:
            Xa += torch.normal(
                torch.tensor(0.0, device=self.device),
                normal_noise * torch.where(Xa == -1, 0, Xa),
                generator=self.generator,
            )

        if inplace:
            batch["Amplitudes"] = Xa
            batch["Polarities"] = Xp
        else:
            batch_copy = deepcopy(batch)
            batch_copy["Amplitudes"] = Xa
            batch_copy["Polarities"] = Xp
            return batch_copy

    @staticmethod
    def predict_angles(outputs, data_format="sdr"):
        if data_format == "sdr":
            sin_theta = outputs[:, ::2]
            cos_theta = outputs[:, 1::2]
        elif data_format == "sin_cos":
            sin_theta = outputs[:, :3]
            cos_theta = outputs[:, 3:]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        angles_deg = torch.rad2deg(angles_rad)  # Converti da radianti a gradi
        angles_deg[:, 0] = angles_deg[:, 0] % 360
        return angles_deg  # Dimensione [batch_size, 3]

    @staticmethod
    def compute_trig_targets(targets, data_format="sdr"):
        # targets è un tensore di dimensione [batch_size, 3] in gradi
        targets_rad = torch.deg2rad(targets)  # Converti da gradi a radianti
        sin_targets = torch.sin(targets_rad)
        cos_targets = torch.cos(targets_rad)
        if data_format == "sdr":
            targets_trig = torch.stack([sin_targets, cos_targets], dim=2)
            targets_trig = targets_trig.flatten(1)
        elif data_format == "sin_cos":
            targets_trig = torch.cat([sin_targets, cos_targets], dim=1)

        return targets_trig  # Dimensione [batch_size, 6]

    @staticmethod
    def zeroing(zeroing_value, X, generator=None, device=None, none_value=0):
        if device is None:
            device = X.device
        elif device != X.device:
            X = X.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Z = (X != none_value).sum(dim=1)
        randomizer = torch.rand(Z.shape, device=device)
        Z_sampler = torch.ceil(Z * randomizer * zeroing_value).to(torch.int32)
        Z_sampler_max = Z_sampler.max().item()
        P = torch.where(X != none_value, 1, 0).float()
        P /= P.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(P, Z_sampler_max, generator=generator)

        rows = torch.arange(Z_sampler.size(0), device=device).repeat_interleave(
            Z_sampler
        )

        # Crea una maschera binaria per selezionare gli elementi
        mask = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_sampler.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        cols = idxs[mask]
        return rows, cols

    @staticmethod
    def zeroing_as_same(zeroing_values, Xp, Xa, generator=None, device=None):
        # Azzera maggiormente le Polarità rispetto alle Ampiezze, ma dove le polarità sono 0, anche le ampiezze sono 0
        assert (
            len(zeroing_values) == 2
        ), "The zeroing_values must be a list of two elements"
        if device is None:
            assert Xp.device == Xa.device, "Xp and Xa must be on the same device"
            device = Xa.device
        elif device != Xa.device or device != Xp.device:
            Xa = Xa.to(device)
            Xp = Xp.to(device)

        if generator is None:
            generator = torch.Generator(device=device)

        Zp = (Xp != 0).sum(dim=1)
        Za = (Xa != -1).sum(dim=1)
        # randomizerP = torch.rand(Zp.shape, device=device)
        # randomizerA = torch.rand(Za.shape, device=device)
        # Z_samplerP = torch.ceil(Zp * randomizerP * zeroing_values[0]).to(torch.int32)
        # Z_samplerA = torch.ceil(Za * randomizerA * zeroing_values[1]).to(torch.int32)
        # Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        # Z_sampler_max = Z_samplerA.max().item()

        # Calcola quanti azzerare tra min e max % delle attive
        min_frac, max_frac = zeroing_values
        Zp_min = torch.floor(Zp * min_frac)
        Zp_max = torch.floor(Zp * max_frac)
        Za_min = torch.floor(Za * min_frac)
        Za_max = torch.floor(Za * max_frac)

        # Interpolazione random tra min e max per ogni riga
        randP = torch.rand_like(Zp, dtype=torch.float)
        randA = torch.rand_like(Za, dtype=torch.float)

        Z_samplerP = (Zp_min + (Zp_max - Zp_min) * randP).to(torch.int32)
        Z_samplerA = (Za_min + (Za_max - Za_min) * randA).to(torch.int32)

        # Sincronizza azzeramenti: ampiezze ≥ polarità
        Z_samplerA = torch.where(Z_samplerP > Z_samplerA, Z_samplerP, Z_samplerA)
        Z_sampler_max = Z_samplerA.max().item()

        assert (
            Z_sampler_max < Za.max().item()
        ), "Z_sampler_max must be less than Za.max().item()"

        probs = torch.where(Xp != 0, 1, 0).float()
        probs /= probs.sum(dim=1, keepdim=True)
        idxs = torch.multinomial(probs, Z_sampler_max, generator=generator)

        rowsP = torch.arange(Z_samplerP.size(0), device=device).repeat_interleave(
            Z_samplerP
        )
        rowsA = torch.arange(Z_samplerA.size(0), device=device).repeat_interleave(
            Z_samplerA
        )

        # Crea una maschera binaria per selezionare gli elementi
        maskP = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerP.unsqueeze(1)
        maskA = torch.arange(idxs.size(1), device=device).expand(
            len(idxs), -1
        ) < Z_samplerA.unsqueeze(1)

        # Usa la maschera per ottenere il risultato finale
        colsP = idxs[maskP]
        colsA = idxs[maskA]
        return rowsP, colsP, rowsA, colsA
