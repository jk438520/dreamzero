# How to run

From `/lute/guests/guest/jakkol/dreamzero`

## Training 
```
./scripts/run_value_function_finetuning.sh
```

## Inference

### Inference server
```
./run_socket_test_optimized_AR.sh
```
### Inference client
```
./scripts/run_eval_value_function.sh
```

### Open-loop evaluation
```bash
./scripts/run_open_loop_yam.sh
./scripts/run_open_loop_droid_vf.sh
./scripts/run_open_loop_droid_vf_long_context.sh
./scripts/run_open_loop_droid_vf_trainstyle.sh
```
