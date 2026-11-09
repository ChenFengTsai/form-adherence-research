#!/usr/bin/env python
"""Build the AI test corpus: generate songs with known target forms via YuE's
infer.py (CommandLineGenerator), and record them in corpus.json.

prompts.json:
  [{"target_form": "ABACA", "genre": "pop upbeat female vocal",
    "lyrics": {"A": ["verse 1 text", "verse 2 text", "verse 3 text"],
               "B": "chorus text", "C": "bridge text"}},
   ...]
  (lyrics may also be a plain string with one blank-line-separated paragraph
   per distinct letter, or already-tagged "[verse]..." text)

    python scripts/generate_corpus.py --prompts configs/prompts.json \
        --infer_cmd configs/yue_cmd.txt --out_dir outputs/corpus --n_per_prompt 3

configs/yue_cmd.txt holds the command template, e.g. (check your infer.py):
  cd external/YuE/inference && python infer.py --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt {genre_file} --lyrics_txt {lyrics_file} --run_n_segments {n_segments} \
    --stage2_batch_size 4 --output_dir {out_dir} --max_new_tokens 3000 --seed {seed}

Re-running resumes: (prompt, seed) pairs already in corpus.json are skipped.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--infer_cmd", required=True,
                    help="command template, or a file containing it")
    ap.add_argument("--out_dir", default="outputs/corpus")
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model_name", default="YuE", help="recorded in corpus.json")
    args = ap.parse_args()

    from formadherence.integration.generators import CommandLineGenerator

    tpl = open(args.infer_cmd).read().strip() if os.path.exists(args.infer_cmd) \
        else args.infer_cmd
    gen = CommandLineGenerator(tpl, args.out_dir)
    with open(args.prompts) as f:
        prompts = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = os.path.join(args.out_dir, "corpus.json")
    corpus = json.load(open(corpus_path)) if os.path.exists(corpus_path) else []
    done = {(c["prompt_index"], c["seed"]) for c in corpus}

    for pi, p in enumerate(prompts):
        for k in range(args.n_per_prompt):
            seed = args.seed0 + k
            if (pi, seed) in done:
                continue
            t0 = time.time()
            try:
                song = gen.generate(p["target_form"], p["lyrics"], p.get("genre", ""), seed=seed)
            except Exception as e:
                print(f"[fail] prompt {pi} seed {seed}: {type(e).__name__}: {e}")
                continue
            corpus.append({"audio": os.path.abspath(song.meta["audio_path"]),
                           "target_form": p["target_form"], "genre": p.get("genre", ""),
                           "seed": seed, "prompt_index": pi, "model": args.model_name,
                           "lyrics": song.meta["lyrics"]})
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=2)
            print(f"[ok]   prompt {pi} {p['target_form']} seed {seed} "
                  f"({time.time() - t0:.0f}s) -> {song.meta['audio_path']}")
    print(f"[done] {len(corpus)} songs in {corpus_path}")


if __name__ == "__main__":
    main()
