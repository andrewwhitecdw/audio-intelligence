"""Regression test: MagInstPhaseToGriffinLim must pass the instantaneous-phase
channels to griffinlim() in the documented order.

The channel layout produced by ComplexToMagInstPhase (and consumed everywhere
else) is [mag, cos(theta), sin(theta)]. griffinlim()'s signature is
griffinlim(specgram, init_phase_cos, init_phase_sin, ...). The bug passed
channel 2 (sin) as init_phase_cos and channel 1 (cos) as init_phase_sin,
rotating the initial phase estimate by 90 degrees whenever phase-initialized
Griffin-Lim is used (rand_init=False).

The module hardcodes rand_init=True today, so this test spies on the
griffinlim() call and asserts on the argument values directly.
"""

import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")


def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


# torchaudio and jsonargparse are not required for this transform; stub them
# so the module imports without the heavy/optional dependencies.
class _Identity:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return x


_stub_module("torchaudio")
_stub_module(
    "torchaudio.transforms",
    Spectrogram=_Identity,
    InverseSpectrogram=_Identity,
)
_stub_module("jsonargparse", Namespace=dict)

from A2SB.audio_transforms import transforms  # noqa: E402


def test_mag_inst_phase_channels_passed_in_order():
    n_fft = 8
    transform = transforms.MagInstPhaseToGriffinLim(
        n_fft=n_fft, win_length=n_fft, hop_length=2
    )

    mag = torch.full((n_fft // 2 + 1, 3), 0.5)
    cos = torch.full((n_fft // 2 + 1, 3), 0.6)
    sin = torch.full((n_fft // 2 + 1, 3), 0.8)
    mag_inst_phase = torch.stack([mag, cos, sin])

    with mock.patch.object(transforms, "griffinlim") as spy:
        spy.return_value = torch.zeros(1)
        transform.forward(mag_inst_phase)

    assert spy.call_count == 1
    args, _kwargs = spy.call_args
    specgram, init_phase_cos, init_phase_sin = args[0], args[1], args[2]

    assert torch.equal(specgram, mag), "specgram must be the magnitude channel"
    assert torch.equal(init_phase_cos, cos), (
        "init_phase_cos must receive channel 1 (cos); "
        "got the sin channel (cos/sin swapped)"
    )
    assert torch.equal(init_phase_sin, sin), (
        "init_phase_sin must receive channel 2 (sin); "
        "got the cos channel (cos/sin swapped)"
    )
