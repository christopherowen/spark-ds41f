. $HOME/spark-lab/agents/candidate-48-own-marks/batches/env.sh
EXP=$B/src/pr-973/experiments/2026-11-08-chat-template-multiturn-fastpath
W=$B/work/pr-973; mkdir -p $W; chmod 777 $W
for b in 1 2; do
  # README A/B bench: 1107 release image (baseline) then candidate image, same cpuset as the PR
  $B/run.sh 973 bench-baseline1107-boot$b $R -- flock $L docker run --rm --name c48om-973-base-b$b --cpuset-cpus=5-9,15-19 --entrypoint python3 \
    -v /home/chris/.cache/huggingface:/home/chris/.cache/huggingface:ro -v $EXP:/exp:ro -v $W:/out \
    $R /exp/bench_chat_template_multiturn.py --iters 8 --label baseline-boot$b --output /out/bench-baseline1107-boot$b.json
  $B/run.sh 973 bench-candidate48-boot$b $C -- flock $L docker run --rm --name c48om-973-cand-b$b --cpuset-cpus=5-9,15-19 --entrypoint python3 \
    -v /home/chris/.cache/huggingface:/home/chris/.cache/huggingface:ro -v $EXP:/exp:ro -v $W:/out \
    $C /exp/bench_chat_template_multiturn.py --iters 8 --label candidate-boot$b --output /out/bench-candidate48-boot$b.json
done
for b in 1 2; do
  # README unit/contract suite on the candidate image (patched files already in the image)
  $B/run.sh 973 pytest-candidate48-boot$b $C -- flock $L docker run --rm --name c48om-973-pyt-b$b --cpuset-cpus=5-9,15-19 --entrypoint pytest \
    -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x:/opt/release_tests -v $RT:/opt/release_tests:ro \
    -w /opt/spark-ds41f/candidate/vllm $C \
    -p no:cacheprovider -p tblib_stub -p offline_skip -q \
    tests/tokenizers_/test_deepseek_v41.py tests/tokenizers_/test_deepseek_v41_effort.py tests/tokenizers_/test_deepseek_v4.py \
    tests/parser/engine/test_deepseek_v4.py tests/parser/engine/test_deepseek_v41.py \
    tests/entrypoints/openai/test_reasoning_enable_thinking.py tests/entrypoints/openai/test_tool_calls_serialization.py
done
