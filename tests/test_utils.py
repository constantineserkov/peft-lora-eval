import pytest
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils import set_seed


@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset seeds before each test to ensure isolation."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)


def test_set_seed_reproducibility_cpu():
    """Test reproducibility on CPU: random outputs match across runs with same seed."""
    seed = 42

    # Run 1
    set_seed(seed)
    rand1 = random.random()
    np_rand1 = np.random.rand(3)
    torch_rand1 = torch.rand(3)

    # Reset (via fixture)
    # Run 2
    set_seed(seed)
    rand2 = random.random()
    np_rand2 = np.random.rand(3)
    torch_rand2 = torch.rand(3)

    # Assert exact match
    assert rand1 == rand2
    np.testing.assert_array_equal(np_rand1, np_rand2)
    torch.testing.assert_close(torch_rand1, torch_rand2)


def test_set_seed_reproducibility_cuda():
    """Test reproducibility on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed = 42
    device = 'cuda'

    # Run 1
    set_seed(seed)
    torch_cuda1 = torch.rand(3, device=device)

    # Reset (via fixture)
    # Run 2
    set_seed(seed)
    torch_cuda2 = torch.rand(3, device=device)

    torch.testing.assert_close(torch_cuda1, torch_cuda2)

def test_set_seed_different_seeds_produce_different_outputs():
    """Test that different seeds produce different outputs."""
    set_seed(42)
    output1 = random.random()

    set_seed(43)
    output2 = random.random()

    assert output1 != output2

def test_set_seed_dataloader_shuffle_reproducibility():
    """Test dataloader shuffle reproducibility with seed."""
    from torch.utils.data import TensorDataset

    # Simple dataset
    data = torch.arange(10)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    set_seed(42)
    batch1_list = [batch[0].tolist() for batch in loader]

    # Reset
    loader_iter = iter(loader)  # But since shuffle is seeded, need to recreate loader

    # Recreate loader after reset
    set_seed(42)
    loader2 = DataLoader(dataset, batch_size=2, shuffle=True)
    batch2_list = [batch[0].tolist() for batch in loader2]

    assert batch1_list == batch2_list