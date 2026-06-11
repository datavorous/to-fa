import copy
import yaml

_yaml = yaml.safe_load(open(__file__.replace("factory/config.py", "config.yaml")))


def load(experiment=None):
    vllm = copy.deepcopy(_yaml["vllm"])
    run = copy.deepcopy(_yaml["run"])

    if experiment:
        overrides = _yaml["experiments"].get(experiment, {})
        for k, v in overrides.get("vllm", {}).items():
            vllm[k] = v
        for k, v in overrides.get("run", {}).items():
            run[k] = v

    class CFG:
        exp = experiment or "baseline"
        model = vllm["served_name"]
        base_url = run["base_url"]
        concurrency = run["concurrency"]
        seed = run["seed"]
        results_dir = run["results_dir"]
        siso_n = run["siso"]["n"]
        siso_max_tokens = tuple(run["siso"]["max_tokens"])
        siso_temperature = tuple(run["siso"]["temperature"])
        silo_n = run["silo"]["n"]
        silo_max_tokens = tuple(run["silo"]["max_tokens"])
        silo_temperature = tuple(run["silo"]["temperature"])
        load_mode = run.get("load_mode", "concurrent")
        rate_rps = run.get("rate_rps", 4.0)
        sweep_param = run.get("sweep_param", "concurrency")
        sweep_values = run.get("sweep_values", [])
        sweep_base_mode = run.get("sweep_base_mode", "concurrent")
        liso_n = run.get("liso", {}).get("n", 0)
        liso_max_tokens = tuple(run.get("liso", {}).get("max_tokens", [64, 256]))
        lilo_n = run.get("lilo", {}).get("n", 0)
        lilo_max_tokens = tuple(run.get("lilo", {}).get("max_tokens", [2048, 3072]))

    return CFG


CFG = load()
