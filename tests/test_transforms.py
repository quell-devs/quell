import pytest
from quell.transforms import Rescale, KeepCalcifications
import torch

def test_rescale():
    input = torch.rand((5,5,5))
    rescale = Rescale()

    # test that encodes changes input values
    encoded = rescale.encodes(input)
    assert not torch.allclose(input, encoded)

    # test that decodes changes encoded values
    decoded = rescale.decodes(encoded)
    assert not torch.allclose(encoded, decoded)

    # test that decodes recovers encoded input values
    assert torch.allclose(input, decoded)

    # test that float16 fails at decodes
    with pytest.raises(TypeError):
        rescale.decodes(encoded.type(torch.float16))

def test_KeepCalcifications_betaformat():
    # in this format values are of the magnitude by 1e-10
    # need to adjust tolerance on allclose to account for this
    atol = 1e-12

    keep_calcifications = KeepCalcifications(beta_format=True)
    input = torch.rand((5,5,5)) * 0.5e-11
    noisy = torch.rand((5,5,5)) * 0.5e-11

    # ensure at least one voxel is a calcification
    noisy[3,3,3] = keep_calcifications.threshold
    calcification_mask = noisy >= keep_calcifications.threshold

    # check calcification mask is not all zeros
    assert torch.any(calcification_mask)

    output = keep_calcifications.encodes(input, noisy)
    # check that only values at calcifications are changed from noisy
    assert torch.allclose(noisy[calcification_mask], output[calcification_mask],atol=atol)
    assert torch.allclose(input[~calcification_mask], output[~calcification_mask],atol=atol)

    # fail float16
    with pytest.raises(TypeError):
        keep_calcifications.encodes(input.type(torch.float16), noisy.type(torch.float16))


def test_KeepCalcifications_rescaleformat():
    keep_calcifications = KeepCalcifications(beta_format=False)
    input = (torch.rand((5,5,5)) - 0.5) * 10
    noisy = (torch.rand((5,5,5)) - 0.5) * 10

    # ensure at least one voxel is a calcification
    noisy[3,3,3] = keep_calcifications.threshold
    calcification_mask = noisy >= keep_calcifications.threshold

    # check calcification mask is not all zeros
    assert torch.any(calcification_mask)

    output = keep_calcifications.encodes(input, noisy)
    # check that only values at calcifications are changed from noisy
    assert torch.allclose(noisy[calcification_mask], output[calcification_mask])
    assert torch.allclose(input[~calcification_mask], output[~calcification_mask])
    
    # works for float16
    output = keep_calcifications.encodes(input.type(torch.float16), noisy.type(torch.float16))
    noisy=noisy.type(torch.float16)
    assert torch.allclose(noisy[calcification_mask], output[calcification_mask])


def test_KeepCalcifications_manualthreshold():
    # fail no kwargs
    with pytest.raises(ValueError):
         KeepCalcifications()

    keep_calcifications = KeepCalcifications(manual_threshold=0.5)
    input = torch.rand((5,5,5))
    noisy = torch.rand((5,5,5))

    # ensure at least one voxel is a calcification
    noisy[3,3,3] = keep_calcifications.threshold
    calcification_mask = noisy >= keep_calcifications.threshold

    # check calcification mask is not all zeros
    assert torch.any(calcification_mask)

    output = keep_calcifications.encodes(input, noisy)
    # check that only values at calcifications are changed from noisy
    assert torch.allclose(noisy[calcification_mask], output[calcification_mask])
    assert torch.allclose(input[~calcification_mask], output[~calcification_mask])

    # works for float16
    output = keep_calcifications.encodes(input.type(torch.float16), noisy.type(torch.float16))
    noisy=noisy.type(torch.float16)
    assert torch.allclose(noisy[calcification_mask], output[calcification_mask])
    assert not torch.allclose(noisy[~calcification_mask], output[~calcification_mask])
