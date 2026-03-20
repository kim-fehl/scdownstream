#!/usr/bin/env python3

import os
import platform

import yaml

os.environ["MPLCONFIGDIR"] = "./tmp/mpl"
os.environ["NUMBA_CACHE_DIR"] = "./tmp/numba"

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from harmonypy import run_harmony
import symphonypy as sp
from threadpoolctl import threadpool_limits

threadpool_limits(int("${task.cpus}"))

PREFIX = "${prefix}"
H5AD = "${h5ad}"
BATCH_COL = "${batch_col}"
HAS_REFERENCE = bool("${meta2.id ?: ''}")
REFERENCE_MODEL = "reference_model/model.h5ad"
TARGET_SUM = 1e4


def batch_key(adata: ad.AnnData) -> str | None:
    if BATCH_COL in adata.obs.columns and adata.obs[BATCH_COL].nunique() > 1:
        return BATCH_COL
    return None


def pca_components(adata: ad.AnnData) -> int:
    return max(1, min(50, adata.n_obs - 1, adata.n_vars - 1))


def harmony_integrate(reference: ad.AnnData, key: str) -> np.ndarray:
    ref_ho = run_harmony(
        reference.obsm["X_pca"],
        meta_data=reference.obs,
        vars_use=key,
        verbose=False,
    )

    Z = np.asarray(ref_ho.Z_corr)
    if Z.ndim != 2:
        raise ValueError(f"Unexpected harmonypy Z_corr shape: {Z.shape}")
    if Z.shape[0] != reference.n_obs and Z.shape[1] == reference.n_obs:
        Z = Z.T
    elif Z.shape[0] != reference.n_obs:
        raise ValueError(
            f"Unexpected harmonypy Z_corr shape {Z.shape} for {reference.n_obs} cells"
        )

    R = np.asarray(ref_ho.R)
    if R.ndim != 2:
        raise ValueError(f"Unexpected harmonypy R shape: {R.shape}")
    if R.shape[0] == reference.n_obs:
        R_kn = R.T
    elif R.shape[1] == reference.n_obs:
        R_kn = R
    else:
        raise ValueError(
            f"Unexpected harmonypy R shape {R.shape} for {reference.n_obs} cells"
        )

    reference.obsm["X_pca_harmony"] = Z
    reference.uns["harmony"] = {
        "Nr": R_kn.sum(axis=1),
        "C": R_kn @ Z,
        "K": int(ref_ho.K),
        "sigma": np.asarray(ref_ho.sigma).squeeze(),
        "ref_basis_loadings": "PCs",
        "ref_basis_adjusted": "X_pca_harmony",
        "vars_use": key,
        "harmony_kwargs": {},
        "converged": bool(ref_ho.check_convergence(1)),
        "R": R_kn,
    }

    return Z


def build_reference(adata: ad.AnnData) -> tuple[ad.AnnData, pd.DataFrame]:
    reference = adata.copy()
    sc.pp.normalize_total(reference, target_sum=TARGET_SUM)
    sc.pp.log1p(reference)
    reference.var["highly_variable"] = True
    sc.pp.scale(reference, max_value=10)
    sc.pp.pca(reference, n_comps=pca_components(reference), zero_center=False)

    key = batch_key(reference)
    if key:
        emb = harmony_integrate(reference, key)
    else:
        emb = reference.obsm["X_pca"].copy()
        reference.obsm["X_pca_harmony"] = emb

    return reference, pd.DataFrame(emb.round(6), index=adata.obs_names)


def map_query(adata: ad.AnnData, reference: ad.AnnData) -> pd.DataFrame:
    query = adata.copy()
    sc.pp.normalize_total(query, target_sum=TARGET_SUM)
    sc.pp.log1p(query)

    sp.tl.map_embedding(
        query,
        reference,
        key=batch_key(query),
        transferred_primary_basis="X_pca_reference",
        transferred_adjusted_basis="X_emb",
    )

    return pd.DataFrame(query.obsm["X_emb"].round(6), index=adata.obs_names)


adata = sc.read_h5ad(H5AD)

if HAS_REFERENCE:
    reference = sc.read_h5ad(REFERENCE_MODEL)
    emb = map_query(adata, reference)
else:
    reference, emb = build_reference(adata)
    reference.write_h5ad(f"{PREFIX}_model.h5ad")

adata.obsm["X_emb"] = emb.to_numpy()
adata.write_h5ad(f"{PREFIX}.h5ad")
emb.to_pickle(f"X_{PREFIX}.pkl")

versions = {
    "${task.process}": {
        "python": platform.python_version(),
        "anndata": ad.__version__,
        "pandas": pd.__version__,
        "scanpy": sc.__version__,
        "symphonypy": getattr(sp, "__version__", "unknown"),
    }
}

with open("versions.yml", "w") as f:
    yaml.dump(versions, f)
