# CLIP-LoRA with Patch Matching


## How to Run

To start training , simply execute:

```bash
bash run_main.sh
```



### `--patch_matching`
- **Type**: `store_true` (flag)  
- **Default**: `False`  
- **Description**:  
  Enables patch-level matching between image patches and text embeddings. When set, the model uses not just the `[CLS]` token, but also patch-level features and compares them to multiple text vectors per class.

### `--noise_scale`
- **Type**: `float`
- **Default**: `0.02`

- **Description**: 
  Adds Gaussian noise to the *fine-grained text embeddings* used for **patch-level matching**.  
This is especially useful to inject diversity when comparing image patches with class text embeddings.

### `--num_fine_weights`
- **Type**: `int`
- **Default**: `4`

- **Description**: 
  The `--num_fine_weights` argument specifies the number of **fine-grained text embeddings per class** used during **patch-level matching**.
This setting is only relevant when `--patch_matching` is enabled. It allows each image patch to compare itself against **multiple semantic variants** of the same class.

