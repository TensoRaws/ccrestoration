def test_sample() -> None:
    import torch

    print(torch.backends.mps.is_available())
    print(torch.cuda.is_available())
    print(torch.backends.mps.is_available())
