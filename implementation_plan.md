# Add Value Function as Auxiliary Loss

This plan details the process of adding a value function (representing task progress) as an auxiliary loss in the DiT action head. We will append the value function to the `action` modality, which natively supports integrating new dimensions without architectural changes.

## User Review Required

> [!IMPORTANT]
> - The dataset size will be duplicated. We will create a script that reads all parquet files, adds the value function to the `action` array, and saves them to a new specified directory.
> - Value function will be calculated as linear task progress (`frame_index / total_frames_in_episode`).
> - Let me know if the equations for the value function need to be tuned to your specifications!

## Proposed Changes

---

### Data Processing Scripts

#### [NEW] `scripts/data/add_value_function.py`
A new script to duplicate a LeRobot dataset and append a value function to the `action` column.
- Reads `meta/episodes.jsonl` to calculate total frames per episode.
- Iterates over all `.parquet` chunks.
- Computes `value_function = frame_index / (episode_length - 1)` (or similar scaling).
- Appends this 1D value to the `action` array in the parquet file (e.g., expanding from 8 to 9 dims).
- Saves the modified parquet files in a target directory.
- Updates metadata:
  - `meta/info.json`: Expand action shape (e.g., `[8]` -> `[9]`).
  - `meta/modality.json`: Append `"value_function": {"start": 8, "end": 9, ...}` under `action`.
  - Recomputes `meta/stats.json` or modifies the `action` stats.

---

### Configurations

#### [NEW] `groot/vla/configs/data/dreamzero/droid_value_relative.yaml`
A data configuration based on `droid_relative.yaml` but overrides the modality configurations to explicitly use the `action.value_function`:
- Modifies `modality_config_oxe_droid.action.modality_keys` to include `action.value_function`.
- Modifies `transform_oxe_droid.transforms` to apply scaling/normalization (e.g., `q99` or `identity`) to `action.value_function`.
- Appends `action.value_function` to `action_concat_order`.

---

### Training Scripts

#### [NEW] `scripts/train/droid_training_value_finetune.sh`
A copy of `droid_training_lora.sh` configured to point to the new dataset and use our new config `dataset=dreamzero/droid_value_relative`. This ensures no changes to the master training logic, merely configuring it to read the 9th dimensionality and predict it with DiT.

---

### Validation & Client Scripts

#### [NEW] `scripts/inference/eval_value_function.py`
A client script similar to `test_client_AR.py` that validates the model's value predictions:
- Parses the newly generated dataset (`meta/episodes.jsonl` and video backend).
- Iterates through all episodes in the validation split (or entire dataset).
- Streams frames to the `socket_test_optimized_AR.py` inference server using the correct frame schedule.
- Extracts `action.value_function` from the server's output response.
- Creates and saves a plot of the episode: **Ground Truth Value Function vs. Predicted Value Function**.

#### [MODIFY] `socket_test_optimized_AR.py`
Minor modifications if needed to ensure the batch output explicitly unpacks `action.value_function` and passes it successfully through the websocket, although the dynamic `batch_to_dict` seems mostly self-sufficient.

## Open Questions

> [!WARNING]  
> - **Dataset Directory**: Do you have a specific output path where the value-function-injected dataset should reside, or should it just be parameterized by the script arguments?
> - **Value Function Definition**: Should the value function simply be `step_id / total_steps`, or do you strictly need a specific shaping (e.g. gamma-discounted return)?
> - **Test Server Deployment**: Do you have the `socket_test_optimized_AR.py` server running manually on an empty GPU, or should the script manage launching it locally for the tests?

## Verification Plan

### Automated Tests
1. Generate the dataset on a small test shard (`check_parquet.py` equivalent).
2. Start `socket_test_optimized_AR.py` with the newly trained value model checkpoint.
3. Run `eval_value_function.py` to stream a full episode.

### Manual Verification
1. Open the generated plot artifacts (`.png` files) showing the value function trajectory over time. The predicted value function should cleanly trace the linear progress.
