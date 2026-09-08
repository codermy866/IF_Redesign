"""Frozen source-trained teacher audit; evaluation labels used ONLY for metrics.

Exports attention and learned utility baselines without retraining or changing
the previous experiments. The teacher retains its original multimodal context.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from cervix_cogalign.cesl import clinical_matrix
from run_cross_fitted_gain import action_features, ce
import train_ices_backbone as tr

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run(fold, seed, cfg):
    torch.set_num_threads(2)
    source = ROOT / cfg["gain_root"] / fold / f"seed_{seed}"
    out = ROOT / cfg["output"] / "teacher_audits" / fold
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"seed_{seed}.npz"
    if target.exists():
        return
    config = tr.read_config(str(ROOT / "configs/ices_v1_exploratory.json"))
    data = tr.load_fold_data(config, fold)
    base = data["payload"]
    side = np.load(ROOT / cfg["clinical_sidecar"], allow_pickle=False)
    np.testing.assert_array_equal(side["ids"], base["ids"])
    meta = json.loads((source / "diagnostic_reference/complete.json").read_text())
    c, _ = clinical_matrix(base["ages"], side["hpvs"].tolist(), side["tcts"].tolist(), meta["fit_indices"])
    selector_c, _ = clinical_matrix(base["ages"], side["hpvs"].tolist(), side["tcts"].tolist(), data["source_indices"]["train"])
    model = tr.create_model(config, data, torch.device("cpu"))
    model.load_state_dict(torch.load(source / "diagnostic_reference/checkpoint.pt", map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    values = joblib.load(source / "value_models.joblib")
    projection = torch.from_numpy(values["projection"])
    scaler, regressor = values["selectors"]["gain_oof"]
    indices = np.array(data["source_indices"]["val"] + data["parts"]["test"]["index"].astype(int).tolist())
    assert not set(indices) & (set(meta["fit_indices"]) | set(meta["cal_indices"]))
    output = {k: [] for k in ("true_gain", "attention", "predicted_gain", "full_probability", "clinical_non_oct_probability", "attention_budget4_probability", "gain_budget4_probability")}
    for i in indices:
        x = base["cluster_features"][i:i+1]
        av = base["available"][i:i+1].bool()
        clinical = torch.from_numpy(c[i:i+1])
        initial = model(x, clinical, av)
        mask = av.clone(); mask[:,25:] = False
        state = x * mask.unsqueeze(-1)
        p0 = model(state, clinical, mask)["logit"].sigmoid()
        slots = torch.arange(25,37)
        xx = state.repeat(12,1,1); mm = mask.repeat(12,1)
        xx[torch.arange(12), slots] = x[0,slots]; mm[torch.arange(12),slots] = True
        pp = model(xx, clinical.repeat(12,1), mm)["logit"].sigmoid().numpy()
        # Label is consulted only after all predictions/action scores are made.
        z = action_features(state, torch.from_numpy(selector_c[i:i+1]), mask, p0, slots, projection, x[0,slots])
        predicted = regressor.predict(scaler.transform(z))
        attention = initial["attention"][0,25:].numpy()
        for name, scores in (("attention",attention),("gain",predicted)):
            chosen = np.argsort(-scores, kind="stable")[:4] + 25
            keep = mask.clone(); keep[:,chosen] = True
            output[name+"_budget4_probability"].append(float(model(x*keep.unsqueeze(-1),clinical,keep)["logit"].sigmoid()[0]))
        y = float(base["labels"][i])
        output["true_gain"].append(ce(y,float(p0[0]))-ce(y,pp))
        output["attention"].append(attention)
        output["predicted_gain"].append(predicted)
        output["full_probability"].append(float(initial["logit"].sigmoid()[0]))
        output["clinical_non_oct_probability"].append(float(p0[0]))
    np.savez_compressed(target, indices=indices, labels=base["labels"][indices].numpy(),
                        centers=np.array(base["centers"])[indices], **{k:np.array(v) for k,v in output.items()})
    (target.with_suffix(".json")).write_text(json.dumps(dict(fold=fold,seed=seed,n=len(indices),
        interpretation="Frozen multimodal teacher baselines; distinct input task from OCT+clinical VLM. Baseline ranking is one-step utility, not the prior sequential policy.",
        utility="Evaluation CE reduction under source-only teacher; model-based surrogate, not clinical ground truth"), indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--fold",required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--config",default="configs/dea_v1.json"); a=p.parse_args()
    run(a.fold,a.seed,json.loads((ROOT/a.config).read_text()))
