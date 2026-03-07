import pytest
import torch
import config

from models import MODEL_REGISTRY
from preprocess import preprocess_for_training, extract_mel_spectrogram


@pytest.mark.parametrize(
    "model_name",
    [
        "ClassificationCNN",
        "DetectionCNN",
        "WaveformClassificationCNN",
        "ClassificationLSTM",
    ],
)
def test_model_forward_pass(model_name, monkeypatch):
    monkeypatch.setattr(config, "MODEL_NAME", model_name)

    # Determine mel/waveform mode
    if model_name in config.WAVEFORM_ONLY_MODELS:
        use_mel = False
    elif model_name in config.MEL_ONLY_MODELS:
        use_mel = True
    else:
        use_mel = config.USE_MEL

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        use_mel=use_mel,
    )

    model.eval()

    # Create a fake waveform batch
    batch_size = 4
    T = int(config.ACOUSTIC_SR * config.SAMPLE_SECONDS)
    x = torch.randn(batch_size, config.IN_CHANNELS, T)

    # Preprocess
    if use_mel:
        x = preprocess_for_training(x, use_mel=True)
    else:
        x = preprocess_for_training(x, use_mel=False)

    # Forward pass
    with torch.inference_mode():
        logits = model(x)

    # Validate output shape
    assert logits.shape == (batch_size, config.NUM_CLASSES)

    # Validate logits are finite
    assert torch.isfinite(logits).all()
