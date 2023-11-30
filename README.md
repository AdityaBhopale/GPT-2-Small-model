# GPT-2 model

# GPT-2 Architecture Modification and Training

This repository contains the implementation of the GPT-2 model along with modifications and enhancements as specified in the given tasks. The tasks include implementing the GPT-2 model, making architectural changes, and implementing a training loop with different parallelization strategies.

## Task 1: GPT-2 Model Implementation and Checkpoints

### Implementation

The GPT-2 model has been implemented from scratch in Python using PyTorch. The model includes key components such as multi-head self-attention, feed-forward networks, and positional encoding. The code is organized in a modular way, following the original GPT-2 design.

### Checkpoint Loading and Validation

The implemented GPT-2 model has been validated by loading the checkpoints of the original GPT-2 125M model. Sample predictions have been performed to ensure the correct functioning of the model.

## Task 2: Transformer Architectural Changes

### Rotary Positional Embedding

The GPT-2 model has been extended to incorporate Rotary Positional Embeddings as specified in Su et al.'s RoFormer. The code has been updated to replace the original positional embeddings with rotary embeddings.

### Group Query Attention

An implementation of the Group Query Attention mechanism has been added to the GPT-2 model. This mechanism, as described in Ainslie et al.'s GQA paper, modifies the model's operation compared to the standard attention mechanism.

### Sliding Window Attention

The GPT-2 model now includes the Sliding Window Attention mechanism based on the insights from Beltagy et al.'s Longformer. The effects of this mechanism on model performance have been observed and analyzed.



### Fully Sharded Data Parallel (FSDP)

The Fully Sharded Data Parallel (FSDP) strategy has been implemented as part of the training loop. This approach shards the model parameters, gradients, and optimizer state to enable training GPT-2-like models on a single machine.


