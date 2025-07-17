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
  Enables patch-level matching between image patches and text embeddings. 

### `--noise_scale`
- **Type**: `float`
- **Default**: `0.02`

- **Description**: 
  Adds Gaussian noise to the *fine-grained text embeddings* used for **patch-level matching**.  


### `--num_fine_weights`
- **Type**: `int`
- **Default**: `4`

- **Description**: 
  The `--num_fine_weights` argument specifies the number of **fine-grained text embeddings per class** used during **patch-level 

