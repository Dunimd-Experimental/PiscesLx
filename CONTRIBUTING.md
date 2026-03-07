<div align="center">

<h1 style="display: flex; flex-direction: column; align-items: center; gap: 12px; margin-bottom: 8px;">
  <span style="display: flex; align-items: center; gap: 12px;">PiscesL1</span>
  <span style="font-size: 0.6em; color: #666; font-weight: normal;">Contributing Guide</span>
</h1>

</div>

First off, thank you for considering contributing to PiscesL1! It's people like you that make PiscesL1 such a great multimodal large language model.

This document provides guidelines and instructions for contributing to the PiscesL1 project. By participating, you are expected to uphold this code and help us maintain a welcoming and productive community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Running the Project](#running-the-project)
  - [Code Style](#code-style)
  - [Commit Messages](#commit-messages)
- [Project Structure](#project-structure)
- [Community](#community)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

- Make sure you have a [GitHub account](https://github.com/signup/free)
- Fork the repository on GitHub
- Set up your development environment (see [Development Guidelines](#development-guidelines))
- Familiarize yourself with the [project structure](#project-structure)

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/mf2023/piscesl1/issues) to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:

#### Before Submitting a Bug Report

- **Check the documentation** for information that might help
- **Check if the bug has already been reported** by searching on GitHub under [Issues](https://github.com/mf2023/piscesl1/issues)
- **Determine which repository the problem should be reported in**

#### How to Submit a Good Bug Report

Bugs are tracked as [GitHub issues](https://github.com/mf2023/piscesl1/issues). Create an issue and provide the following information:

- **Use a clear and descriptive title** for the issue to identify the problem
- **Describe the exact steps to reproduce the problem** in as many details as possible
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed** and why it's a problem
- **Explain which behavior you expected to see instead and why**
- **Include code samples and error logs** which show the problem

**Example:**

```markdown
**Description:**
Training fails with CUDA out of memory error on 7B model

**Steps to Reproduce:**
1. Run: python manage.py train --model_size 7B --dataset Chinese2
2. Observe the error after first epoch

**Expected Behavior:**
Training should complete successfully with default settings

**Actual Behavior:**
RuntimeError: CUDA out of memory

**Environment:**
- OS: Ubuntu 22.04
- GPU: NVIDIA RTX 3090 (24GB)
- Python: 3.11
- PyTorch: 2.1.0
- PiscesL1 VERSION: 1.0.0
- PiscesL1 CVERSION: 0.3.1
```

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/mf2023/piscesl1/issues). When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title** for the issue to identify the suggestion
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful** to most PiscesL1 users
- **List some other LLM frameworks where this enhancement exists**

### Pull Requests

1. Fork the repo and create your branch from `master`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the code compiles without errors or warnings
5. Make sure your code follows the style guidelines
6. Issue that pull request!

#### Pull Request Process

1. Update the CHANGELOG with details of changes if applicable
2. Update the README with details of changes to the interface if applicable
3. The PR will be merged once you have the sign-off of at least one maintainer

## Development Guidelines

### Setting Up Development Environment

#### Prerequisites

- **Python** (3.10+): [Install Python](https://www.python.org/downloads/)
- **PyTorch** (2.0+): [Install PyTorch](https://pytorch.org/get-started/locally/)
- **CUDA** (11.8+ recommended for GPU support): [CUDA Download](https://developer.nvidia.com/cuda-downloads)
- **Git**: [Install Git](https://git-scm.com/downloads)

#### Clone the Repository

```bash
git clone https://github.com/mf2023/piscesl1.git
cd piscesl1
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Verify Installation

```bash
python manage.py check --gpu --dependencies
```

### Running the Project

#### Start Inference Service

```bash
python manage.py serve --model_size 7B --port 8000
```

#### Run Training

```bash
python manage.py train --model_size 7B --dataset Chinese2
```

#### Run Benchmarks

```bash
python manage.py benchmark --benchmark mmlu --model checkpoint.pt
```

#### Check System Status

```bash
python manage.py check --gpu
python manage.py monitor
```

### Code Style

#### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```bash
# Format with black
black .

# Lint with flake8
flake8 .

# Type check with mypy
mypy .
```

#### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes (model/) | YvXxxXxx | `YvAttention`, `YvMoELayer` |
| Classes (opss/) | POPSSXxxXxx | `POPSSOrchestrator` |
| Classes (other) | PiscesLxXxxXx | `PiscesL1Config` |
| Functions | snake_case | `load_checkpoint()` |
| Variables | snake_case | `hidden_size` |
| Constants | UPPER_SNAKE_CASE | `MAX_SEQUENCE_LENGTH` |
| Private members | _leading_underscore | `_internal_state` |

#### Documentation

- All public classes and functions must have docstrings
- Use Google-style docstrings
- Documentation should include examples where appropriate

```python
def train_model(config: PiscesL1Config, dataset: str) -> None:
    """Train the PiscesL1 model with the given configuration.
    
    Args:
        config: Model configuration object containing hyperparameters.
        dataset: Name of the dataset to use for training.
    
    Raises:
        ValueError: If the dataset is not found.
        RuntimeError: If GPU memory is insufficient.
    
    Example:
        >>> config = PiscesL1Config(hidden_size=4096)
        >>> train_model(config, "Chinese2")
    """
    pass
```

### Commit Messages

This project uses **date-based commit messages** in the format `YYYY.MM.DD`:

```
2026.03.07
```

#### Format

- Use the **current date** in `YYYY.MM.DD` format
- No additional description
- No body or footer

#### Examples

```bash
# Good
git commit -m "2026.03.07"

# Bad - don't use conventional commits or descriptions
git commit -m "feat(attention): add flash attention support"
git commit -m "fix bug in training loop"
```

#### Why Date-Based?

- **Simple**: No need to think about commit message format
- **Clear timeline**: Easy to see when changes were made
- **Consistent**: All commits follow the same pattern
- **Changelog**: Detailed changes are tracked in CHANGELOG

#### Tracking Changes

Since commit messages are minimal, detailed change information is maintained in:

- **CHANGELOG**: Version history and release notes
- **GitHub Issues/PRs**: Detailed discussion and context
- **Code comments**: Inline documentation for complex changes

## Project Structure

```
PiscesL1/
├── model/                    # Model implementations
│   ├── core/                # Core components (attention, embedding, norms)
│   │   ├── attention.py     # Attention mechanisms (Flash, Linear, etc.)
│   │   ├── blocks.py        # Transformer blocks
│   │   ├── cache.py         # KV cache implementations
│   │   ├── embedding.py     # Position and token embeddings
│   │   ├── hybrid.py        # Hybrid Attention-SSM architecture
│   │   ├── mamba3.py        # Mamba-3 SSM implementation
│   │   ├── model.py         # Main model class
│   │   └── norms.py         # Normalization layers
│   ├── moe/                 # Mixture of Experts
│   │   ├── expert.py        # Expert network implementations
│   │   ├── gate.py          # MoE routing gates
│   │   └── layer.py         # MoE layer
│   ├── multimodal/          # Multimodal components
│   │   ├── vision.py        # Vision encoder/decoder
│   │   ├── audio.py         # Audio encoder
│   │   ├── video.py         # Video encoder
│   │   ├── fusion.py        # Multimodal fusion
│   │   └── agentic.py       # Agentic capabilities
│   ├── reasoning/           # Reasoning modules
│   │   └── reasoner/        # Chain-of-thought, multipath reasoning
│   ├── generation/          # Text generation
│   │   ├── sampler.py       # Sampling strategies
│   │   └── speculative.py   # Speculative decoding
│   ├── tokenizer/           # Tokenization
│   └── utils/               # Model utilities
├── opss/                     # Operations and infrastructure
│   ├── train/               # Training operations
│   │   ├── impl.py          # Training implementation
│   │   ├── checkpoint.py    # Checkpoint management
│   │   ├── fsdp.py          # Fully Sharded Data Parallel
│   │   └── lr_scheduler.py  # Learning rate schedulers
│   ├── infer/               # Inference operations
│   │   ├── sampling.py      # Sampling implementations
│   │   └── speculative.py   # Speculative decoding
│   ├── quantize/            # Quantization
│   │   ├── engine.py        # Quantization engine
│   │   └── methods.py       # Quantization methods
│   ├── mcp/                 # Model Context Protocol
│   ├── agents/              # Agent implementations
│   │   ├── cmu/             # Computer Use Agent
│   │   └── experts/         # Expert agents
│   ├── watermark/           # Watermarking
│   └── run/                 # Run management
├── tools/                    # Tool implementations
│   ├── train/               # Training tools
│   ├── infer/               # Inference tools
│   ├── benchmark/           # Benchmarking tools
│   ├── data/                # Data processing tools
│   ├── monitor/             # Monitoring tools
│   └── wmc/                 # Watermark checking
├── configs/                  # Configuration files
│   ├── model/               # Model configurations (0.5B, 7B, 70B, etc.)
│   ├── train/               # Training configurations
│   ├── version.py           # Version information
│   └── dataset.yaml         # Dataset configuration
├── utils/                    # Utility functions
│   ├── dc.py                # Logging utilities
│   └── paths.py             # Path management
├── manage.py                 # CLI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project readme
├── SECURITY.md               # Security policy
├── CODE_OF_CONDUCT.md        # Code of conduct
└── LICENSE                   # Apache 2.0 License
```

## Model Contribution Guidelines

### Adding a New Model Component

1. Create a new file under `model/<category>/` or add to an existing category
2. Follow the naming convention (YvXxxXxx for model/, POPSSXxxXxx for opss/)
3. Add comprehensive docstrings
4. Register the component if needed
5. Add unit tests

### Example: Adding a New Attention Mechanism

```python
# model/core/attention.py

class YvFlashAttention2(nn.Module):
    """FlashAttention-2 implementation with memory-efficient attention.
    
    This implementation uses the FlashAttention-2 algorithm for
    memory-efficient attention computation with O(N) memory complexity.
    
    Args:
        hidden_size: Hidden dimension size.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    
    Example:
        >>> attn = YvFlashAttention2(4096, 32)
        >>> output = attn(query, key, value)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Implementation here
        pass
```

### Adding a New Training Mode

1. Create configuration in `configs/train/`
2. Implement training logic in `opss/train/`
3. Update `manage.py` to support the new mode
4. Add documentation

## Community

### Communication Channels

- **Gitee Issues** (Primary): Bug reports, feature requests - https://gitee.com/dunimd/piscesl1
- **GitHub Issues** (Mirror): Alternative access - https://github.com/mf2023/piscesl1/issues
- **GitHub Discussions**: For questions and community interaction
- **ModelScope**: Model weights - https://www.modelscope.cn/models/mfchina2024/PiscesL1

### Repositories

- **Gitee** (Primary): https://gitee.com/dunimd/piscesl1.git
- **GitHub** (Mirror): https://github.com/mf2023/piscesl1.git

### Recognition

Contributors will be recognized in our CHANGELOG and release notes.

## License

By contributing to PiscesL1, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

Thank you for contributing to PiscesL1!
