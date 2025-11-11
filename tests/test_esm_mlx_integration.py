#!/usr/bin/env python3
"""
Test script for ESM-MLX integration with OpenFold-3-MLX.

This script demonstrates how to use ESM-MLX as a direct replacement
for MSA features in the OpenFold-3-MLX pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add OpenFold3 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openfold3.core.model.feature_embedders.esm_mlx_embedder import (
        ESMMLXEmbedder,
        ESMMLXInputEmbedder
    )
    print("✅ ESM-MLX embedder imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ESM-MLX embedder: {e}")
    sys.exit(1)

try:
    import esm_mlx
    print(f"✅ ESM-MLX available: version {esm_mlx.__version__}")
except ImportError:
    print("❌ ESM-MLX not available. Install with: pip install esm-mlx")
    sys.exit(1)


def create_test_batch(sequence: str, batch_size: int = 1):
    """Create a test batch dictionary for OpenFold3."""

    # Convert sequence to amino acid indices
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }

    N_token = len(sequence)
    device = torch.device('cpu')

    # Convert sequence to indices
    aatype = torch.tensor([aa_to_idx.get(aa, 20) for aa in sequence], device=device)
    aatype = aatype.unsqueeze(0).expand(batch_size, -1)  # [batch_size, N_token]

    # Create one-hot encoded residue types
    restype = torch.nn.functional.one_hot(aatype, num_classes=21).float()  # [batch_size, N_token, 21]

    # Create dummy profile (uniform distribution over amino acids)
    profile = torch.full((batch_size, N_token, 22), 1.0/22, device=device)

    # Create token indices
    token_index = torch.arange(N_token, device=device).unsqueeze(0).expand(batch_size, -1)

    batch = {
        'aatype': aatype,
        'restype': restype,
        'profile': profile,
        'token_index': token_index,
    }

    return batch


def test_esm_mlx_embedder():
    """Test ESMMLXEmbedder functionality."""

    print("\n🧪 Testing ESMMLXEmbedder...")

    # Test sequence (ubiquitin - small, well-studied protein)
    sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    print(f"Test sequence: {sequence[:30]}... (length: {len(sequence)})")

    # Create test batch
    batch = create_test_batch(sequence)

    # Model parameters
    c_m_feats = 34
    c_m = 256
    c_s_input = 44  # 21 (restype) + 22 (profile) + 1 (deletion_mean placeholder)

    # Add deletion_mean placeholder to s_input
    batch_size, N_token = batch['restype'].shape[:2]
    deletion_mean = torch.zeros(batch_size, N_token, 1)
    s_input = torch.cat([batch['restype'], batch['profile'], deletion_mean], dim=-1)

    try:
        # Initialize ESM-MLX embedder
        embedder = ESMMLXEmbedder(
            c_m_feats=c_m_feats,
            c_m=c_m,
            c_s_input=c_s_input,
            esm_model_size="small",  # Use small model for faster testing
            use_quantization=True,
            num_virtual_msa_seqs=8,  # Fewer sequences for testing
        )

        print(f"✅ ESM-MLX embedder initialized")

        # Run forward pass
        print("🔄 Running forward pass...")
        m, msa_mask = embedder(batch, s_input)

        print(f"✅ Forward pass successful!")
        print(f"   MSA features (m): {m.shape}")
        print(f"   MSA mask: {msa_mask.shape}")
        print(f"   MSA features mean: {m.mean().item():.6f}")
        print(f"   MSA features std: {m.std().item():.6f}")

        return True

    except Exception as e:
        print(f"❌ ESMMLXEmbedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_esm_mlx_input_embedder():
    """Test complete ESMMLXInputEmbedder functionality."""

    print("\n🧪 Testing ESMMLXInputEmbedder...")

    sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    batch = create_test_batch(sequence)

    # Model parameters
    c_s_input = 43  # Will be adjusted internally
    c_s = 384
    c_z = 128
    c_m = 256

    try:
        # Initialize complete input embedder
        input_embedder = ESMMLXInputEmbedder(
            c_s_input=c_s_input,
            c_s=c_s,
            c_z=c_z,
            c_m=c_m,
            esm_model_size="small",
            use_quantization=True,
            num_virtual_msa_seqs=8,
        )

        print(f"✅ ESM-MLX input embedder initialized")

        # Run forward pass
        print("🔄 Running complete forward pass...")
        s_input, s, z, m, msa_mask = input_embedder(batch)

        print(f"✅ Complete forward pass successful!")
        print(f"   s_input: {s_input.shape}")
        print(f"   s: {s.shape}")
        print(f"   z: {z.shape}")
        print(f"   m: {m.shape}")
        print(f"   msa_mask: {msa_mask.shape}")

        # Verify shapes are compatible with OpenFold3 expectations
        batch_size, N_token = s.shape[:2]
        N_msa = m.shape[-3]

        assert s.shape == (batch_size, N_token, c_s), f"s shape mismatch: {s.shape} != ({batch_size}, {N_token}, {c_s})"
        assert z.shape == (batch_size, N_token, N_token, c_z), f"z shape mismatch: {z.shape}"
        assert m.shape == (batch_size, N_msa, N_token, c_m), f"m shape mismatch: {m.shape}"

        print(f"✅ All output shapes are compatible with OpenFold3!")

        return True

    except Exception as e:
        print(f"❌ ESMMLXInputEmbedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_esm_mlx_performance():
    """Benchmark ESM-MLX performance on different sequence lengths."""

    print("\n⏱️  Benchmarking ESM-MLX performance...")

    import time

    sequences = {
        "Short (50 AA)": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLE",
        "Medium (100 AA)": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVEPSDTIEN",
        "Long (200 AA)": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG" * 3,
    }

    # Initialize embedder once
    embedder = ESMMLXInputEmbedder(
        c_s_input=43, c_s=384, c_z=128, c_m=256,
        esm_model_size="small", use_quantization=True, num_virtual_msa_seqs=4
    )

    for name, sequence in sequences.items():
        print(f"\n📏 Testing {name}: {len(sequence)} residues")

        batch = create_test_batch(sequence)

        # Warmup
        try:
            with torch.no_grad():
                _ = embedder(batch)
        except Exception as warmup_error:
            print(f"   ❌ Warmup failed: {warmup_error}")
            continue

        # Benchmark
        try:
            start_time = time.time()
            with torch.no_grad():
                s_input, s, z, m, msa_mask = embedder(batch)
            end_time = time.time()
        except Exception as benchmark_error:
            print(f"   ❌ Benchmark failed: {benchmark_error}")
            import traceback
            traceback.print_exc()
            continue

        runtime = end_time - start_time
        residues_per_second = len(sequence) / runtime

        print(f"   Runtime: {runtime:.3f}s")
        print(f"   Speed: {residues_per_second:.0f} residues/second")
        print(f"   Memory usage: ~{torch.cuda.memory_allocated() / 1e6:.1f} MB" if torch.cuda.is_available() else "   Memory usage: CPU only")


if __name__ == "__main__":
    print("🚀 ESM-MLX Integration Test Suite")
    print("=" * 50)

    success_count = 0
    total_tests = 2

    # Test individual ESM-MLX embedder
    if test_esm_mlx_embedder():
        success_count += 1

    # Test complete input embedder
    if test_esm_mlx_input_embedder():
        success_count += 1

    # Optional performance benchmark
    try:
        benchmark_esm_mlx_performance()
    except Exception as e:
        print(f"⚠️  Benchmark failed: {e}")

    # Summary
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("🎉 All tests passed! ESM-MLX integration is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Integrate ESM-MLX embedder into OpenFold3 model configuration")
        print("2. Test with full OpenFold3 pipeline using --use-esm-mlx flag")
        print("3. Compare performance vs traditional MSA-based approach")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1)