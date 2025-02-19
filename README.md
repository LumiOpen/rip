# RIP prompt generation and filtering

Implementation of 
[R.I.P.: Better Models by Survival of the Fittest Prompts](https://arxiv.org/abs/2501.18578)

This is designed to be run in a highly parallel fashion using SLURM, but could
be easily adapted to other workflows.

## Directions

This is a 3 stage process.

1. Generate 64 responses for each prompt in a prompt dataset.
2. Rate all the responses using a reward model.
3. Filter out prompts with large response variance.


## Stage 1

Script: `lumi_scripts/launch_generate_standard.sh`

Launch the dispatcher server and as many rip processes as you need.  The rip
processes will use VLLM to inference the model, and if you want to limit the
number of GPUs in use by each process, you'll need to control the GPUs that
VLLM can see with `CUDA_DEVICES_VISIBLE`, because VLLM doesn't seem to have an
option to control that directly.

## Stage 2

Script: `lumi_scripts/launch_score_standard.sh`

Execution is roughly the same as the last mode.

Some caveats:

The script is hardcoded to work with ArmoRM, and extracts ArmoRMs scoring
objectives and coefficients in addition to the final score.  For this reason
it uses transformers instead of VLLM, so is a bit slower, but it's an 8B
model and doing almost entirely prefill, so it should go fairly quickly.

ArmoRM only handles sequence lengths up to 4K.  We give everything longer than
this invalid scores. This is probably not ideal, and might result in us
throwing out good samples during the next Stage; this could be revisited.

## Stage 3

Filter the final file. No batch processing is necessary for this.  This could
go OOM with a large prompt dataset, but I haven't run into that problem yet.
The script could be easily made more efficient here.

```bash
python rip.py filter --input_path scored.jsonl --accepted_path accepted.jsonl --rejected_path rejected.jsonl
```

The accepted rows will end up in the `accepted.jsonl` and the rejected rows
will be in the `rejected.jsonl`

You can easily reformat the final jsonl output into a more typical multi turn
chat format in jsonl like this:

```bash
cat accepted.jsonl | jq -c '{"messages": [{"role": "user", "content": .prompt}, {"role": "assistant", "content": .best_response.response }]}' > accepted_reformatted.jsonl
```



