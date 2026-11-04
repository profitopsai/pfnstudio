# pfnstudio — CLI

The command-line interface for [PFN Studio](https://github.com/profitopsai/pfnstudio),
the toolkit for training [prior-fitted foundation models](https://arxiv.org/abs/2112.10510).

## Install

```bash
pip install pfnstudio
```

Run a pretrained base model — `--install-deps` installs what it needs on the
first run (PyTorch when missing, then the model's own packages):

```bash
pip install pfnstudio
pfnstudio run cdfm-base --install-deps
```

For training (requires PyTorch):

```bash
pip install "pfnstudio[torch]"
```

## Commands

```text
pfnstudio init <dir>            # scaffold a new FM project
pfnstudio validate <path>       # check artifacts against JSON Schema
pfnstudio lint <project>        # cross-reference + style checks
pfnstudio sample <prior.yaml>   # draw N tasks from a prior
pfnstudio run <run.yaml>        # execute a training run end-to-end
pfnstudio predict <run-dir>     # inference against a trained checkpoint
pfnstudio export <project>     # tar-gzipped project archive
```

Base models from the marketplace, Ollama-shaped — the CLI is a client of a
local daemon that hosts the models:

```text
pfnstudio run <slug>[:tag]      # pull, pin (lock.json + digest), load into the daemon, run its sample
                                #   --install-deps: PyTorch + the model's packages, on first run
pfnstudio pull <slug>[:tag]     # fetch the adapter bundle + checkpoint into the local store, no run
pfnstudio ls                    # what is in the local store (slug, tag, digest, sizes, loaded?)
pfnstudio ps                    # what the daemon has loaded right now
pfnstudio show <slug>           # listing, lock, tasks (capabilities) of one model
pfnstudio rm <slug>[:tag]       # delete it from the store (asks first; shared weights are kept)
pfnstudio serve                 # run the daemon in the foreground (`run` starts it on demand)
pfnstudio create <name> -f Pfnfile   # compose a model in the studio FROM a base model; `run <name>` uses it
```

`pfnstudio serve <run.yaml> --checkpoint <dir>` is the single-model form the
model container and the studio's Deployments use.

Run `pfnstudio --help` for the full list and `<cmd> --help` for each
subcommand's flags.

## What this CLI is for

PFN Studio organises every PFN project around five first-class
artifacts: **priors** (synthetic data generators), **models** (block
compositions), **evals** (benchmarks + metrics), **runs** (training
manifests), and **initiatives** (research workstreams). This CLI
operates on the file layout those artifacts produce — scaffolding new
projects, validating them, running training, and exporting them for
sharing.

The full story (concepts, architecture, examples, marketplace catalog)
lives at the main repo:
**[github.com/profitopsai/pfnstudio](https://github.com/profitopsai/pfnstudio)**

## License

Apache-2.0.
