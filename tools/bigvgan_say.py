"""Offline BigVGAN synthesis from SAM LPC-frame features."""

import argparse
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave


HEADER = struct.Struct("<4sIII")
RECORD = struct.Struct("<IIf")


def require_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise RuntimeError("NumPy is required; install the BigVGAN environment requirements") from exc
    return numpy


def read_lpc_frames(path):
    np = require_numpy()
    records = []
    with open(path, "rb") as f:
        header = f.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("LPC feature file has no complete header")
        magic, version, sample_rate, order = HEADER.unpack(header)
        if magic != b"SLPC" or version != 1 or sample_rate <= 0 or order <= 0:
            raise ValueError("unsupported LPC feature file")
        while True:
            raw = f.read(RECORD.size)
            if not raw:
                break
            if len(raw) != RECORD.size:
                raise ValueError("truncated LPC frame record")
            samples, flags, amp = RECORD.unpack(raw)
            coeffs_raw = f.read(4 * (order + 1))
            if len(coeffs_raw) != 4 * (order + 1):
                raise ValueError("truncated LPC coefficients")
            if samples:
                coeffs = np.frombuffer(coeffs_raw, dtype="<f4").astype(np.float64)
                if not np.all(np.isfinite(coeffs)) or not math.isfinite(amp):
                    raise ValueError("non-finite LPC feature")
                if flags not in (0, 1, 2):
                    raise ValueError("invalid LPC frame flags")
                records.append((samples, flags, amp, coeffs))
    if not records:
        raise ValueError("LPC feature file contains no audio frames")
    return sample_rate, order, records


def load_bigvgan(checkpoint, repo, device):
    sys.path.insert(0, str(repo))
    try:
        import torch
        import bigvgan
    except ImportError as exc:
        raise RuntimeError(
            "BigVGAN dependencies are unavailable; install the requirements from the BigVGAN repository"
        ) from exc

    checkpoint = pathlib.Path(checkpoint)
    if checkpoint.is_dir():
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "bigvgan_generator.pt"
    else:
        weights_path = checkpoint
        config_path = checkpoint.parent / "config.json"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("checkpoint needs config.json and bigvgan_generator.pt")
    h = bigvgan.load_hparams_from_json(str(config_path))
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    checkpoint_data = torch.load(str(weights_path), map_location="cpu")
    weights = checkpoint_data.get("generator", checkpoint_data)
    model.load_state_dict(weights)
    model.remove_weight_norm()
    model.eval().to(device)
    return torch, model, h


def lpc_to_mels(sample_rate, records, h):
    np = require_numpy()
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required; install the BigVGAN environment requirements") from exc

    target_rate = int(h.sampling_rate)
    hop = int(h.hop_size)
    n_fft = int(h.n_fft)
    n_mels = int(h.num_mels)
    if target_rate <= 0 or hop <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("invalid BigVGAN STFT parameters in checkpoint config")
    fmax = getattr(h, "fmax", None)
    if fmax is None:
        fmax = target_rate / 2
    fmax = min(fmax, sample_rate / 2)
    if fmax <= h.fmin:
        raise ValueError("BigVGAN mel range exceeds the source LPC sample rate")
    mel_basis = librosa.filters.mel(
        sr=target_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=h.fmin,
        fmax=fmax,
    ).astype(np.float64)
    frequencies = np.arange(n_fft // 2 + 1, dtype=np.float64) * target_rate / n_fft
    angular = 2.0 * np.pi * frequencies / sample_rate
    powers = np.exp(-1j * angular[:, None] * np.arange(records[0][3].size)[None, :])
    frame_mels = []
    centers = []
    silence_ranges = []
    cursor = 0
    for samples, flags, amp, coeffs in records:
        center = cursor + samples * 0.5
        centers.append(center * target_rate / sample_rate)
        if flags == 0:
            mel = np.full(n_mels, np.log(1e-5), dtype=np.float64)
            silence_ranges.append((cursor, cursor + samples))
        else:
            response = powers @ coeffs
            envelope = 1.0 / np.maximum(np.abs(response), 1e-6)
            envelope *= max(abs(amp), 1e-5)
            mel = np.log(np.maximum(mel_basis @ envelope, 1e-5))
        frame_mels.append(mel)
        cursor += samples

    total_samples = int(round(cursor * target_rate / sample_rate))
    frame_count = max(1, int(math.ceil(total_samples / hop)))
    positions = np.minimum(np.arange(frame_count) * hop + hop * 0.5, total_samples)
    centers = np.asarray(centers, dtype=np.float64)
    frame_mels = np.asarray(frame_mels, dtype=np.float64)
    mel = np.empty((n_mels, frame_count), dtype=np.float32)
    for band in range(n_mels):
        mel[band] = np.interp(positions, centers, frame_mels[:, band]).astype(np.float32)

    linear_mel = np.exp(mel)
    peak = float(linear_mel.max())
    if peak > 0.0:
        linear_mel *= 10.0 / peak
    mel = np.log(np.maximum(linear_mel, 1e-5)).astype(np.float32)
    return mel, target_rate, total_samples, silence_ranges


def write_wav(path, audio, sample_rate, target_samples, source_rate, silence_ranges):
    np = require_numpy()
    audio = audio[:target_samples]
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))
    for start, end in silence_ranges:
        lo = int(start * sample_rate / source_rate)
        hi = min(target_samples, int(end * sample_rate / source_rate))
        audio[lo:hi] = 0.0
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-say", required=True, help="path to the built sam_say executable")
    parser.add_argument("--data", required=True, help="voice data directory")
    parser.add_argument("--voice", default="Sam")
    parser.add_argument("--bigvgan-repo", required=True, help="path to the NVIDIA BigVGAN checkout")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or generator .pt file")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("text", help="text to speak, or @path to a UTF-8 text file")
    parser.add_argument("output", help="output WAV path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="sam-bigvgan-") as temp:
        features = pathlib.Path(temp) / "speech.slpc"
        reference = pathlib.Path(temp) / "sam.wav"
        command = [
            args.sam_say,
            "--data",
            args.data,
            "--voice",
            args.voice,
            "--vocoder",
            "sam",
            "--dump-lpc",
            str(features),
            args.text,
            str(reference),
        ]
        subprocess.run(command, check=True)
        source_rate, _, records = read_lpc_frames(features)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required; install the BigVGAN environment requirements") from exc
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
        torch, model, h = load_bigvgan(args.checkpoint, args.bigvgan_repo, device)
        mel, output_rate, target_samples, silence_ranges = lpc_to_mels(source_rate, records, h)
        features_tensor = torch.from_numpy(mel).unsqueeze(0).to(device)
        with torch.inference_mode():
            generated = model(features_tensor).squeeze().float().cpu().numpy()
        write_wav(args.output, generated, output_rate, target_samples, source_rate, silence_ranges)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bigvgan_say: {exc}", file=sys.stderr)
        raise SystemExit(1)
