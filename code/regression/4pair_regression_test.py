#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# ---------------------------------------------------------------------------------
# 1. Settings & Loadings
# ---------------------------------------------------------------------------------
logging.getLogger("py4j").setLevel(logging.WARN)
logging.getLogger("pyspark").setLevel(logging.WARN)

pair_attr_path = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_similarity_pairs_attributes_50geo.parquet"
output_json_path = "/home/wangyd/Projects/macs_thesis/yangyu/regression_results/regression_models.json"

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

pairs_attr = spark.read.parquet(pair_attr_path)

# environment = not (self OR firstOrderTie OR secondOrderTie)
pairs_attr = pairs_attr.withColumn(
    "environment",
    ~(
        F.coalesce(F.col("self"), F.lit(False)) |
        F.coalesce(F.col("firstOrderTie"), F.lit(False)) |
        F.coalesce(F.col("secondOrderTie"), F.lit(False))
    )
)

pairs_attr.select(
    "src", "dst", "sim", "self", "environment",
    "firstOrderTie", "secondOrderTie", "geoDistance"
).show(20, truncate=False)

# ---------------------------------------------------------------------------------
# 2. Tie-type variable + numeric casting
# ---------------------------------------------------------------------------------
pairs_attr = pairs_attr.withColumn(
    "tie_type",
    F.when(F.col("self") == True, F.lit("self"))
     .when(F.col("firstOrderTie") == True, F.lit("firstOrder"))
     .when(F.col("secondOrderTie") == True, F.lit("secondOrder"))
     .otherwise(F.lit("environment"))
)

bool_cols = [
    "self",
    "environment",
    "firstOrderTie",
    "secondOrderTie",
    "nationalityAffinity",
    "affiliationAffinity",
    "genderAffinity",
    "educationAffinity",
    "religionAffinity",
    "languageAffinity",
]

df_num = pairs_attr
for c in bool_cols:
    df_num = df_num.withColumn(c, F.col(c).cast("double"))

# ---------------------------------------------------------------------------------
# 3. One-hot encode tie_type (manual dummies; environment is implicit baseline)
# ---------------------------------------------------------------------------------
df_num = (
    df_num
    .withColumn("tie_firstOrder",  (F.col("tie_type") == F.lit("firstOrder")).cast("double"))
    .withColumn("tie_secondOrder", (F.col("tie_type") == F.lit("secondOrder")).cast("double"))
    .withColumn("tie_self",        (F.col("tie_type") == F.lit("self")).cast("double"))
)

tie_dummy_cols = ["tie_firstOrder", "tie_secondOrder", "tie_self"]

# ---------------------------------------------------------------------------------
# 4. Select features (including geoDistance)
# ---------------------------------------------------------------------------------
affinity_cols = [
    "birthYearDiff",
    "geoDistance",
    "nationalityAffinity",
    "affiliationAffinity",
    "genderAffinity",
    "educationAffinity",
    "religionAffinity",
    "languageAffinity",
]

graph_cols = [
    "graphNodeCount",
    "graphDensity",
    "avgClustering",
]

base_features_continuous = affinity_cols + graph_cols
base_continuous = ["sim"] + base_features_continuous

df_clean = df_num.select(
    ["sim", "tie_type"] + tie_dummy_cols + base_features_continuous
).dropna()

# ---------------------------------------------------------------------------------
# 5. Normalize numeric variables (z-scores)
# ---------------------------------------------------------------------------------
stats_exprs = []
for c in base_continuous:
    stats_exprs.append(F.mean(c).alias(f"{c}_mean"))
    stats_exprs.append(F.stddev(c).alias(f"{c}_std"))
stats_row = df_clean.select(*stats_exprs).collect()[0]

means = {c: stats_row[f"{c}_mean"] for c in base_continuous}
stds = {c: stats_row[f"{c}_std"] for c in base_continuous}

print("\n=== Mean and Std Used for Z-Score Normalization ===")
for c in base_continuous:
    print(f"{c:30s} mean = {means[c]: .6f}   std = {stds[c]: .6f}" if stds[c] is not None else
          f"{c:30s} mean = {means[c]: .6f}   std = None")

df_norm = df_clean
for c in base_continuous:
    mu = means[c]
    sigma = stds[c]
    if sigma is None or sigma == 0:
        df_norm = df_norm.withColumn(f"{c}_z", F.lit(0.0))
    else:
        df_norm = df_norm.withColumn(f"{c}_z", (F.col(c) - F.lit(mu)) / F.lit(sigma))

# ---------------------------------------------------------------------------------
# 6. Original linear interaction terms
# ---------------------------------------------------------------------------------
df_norm = df_norm.withColumn(
    "geoDistance_x_firstOrder",
    F.col("geoDistance_z") * F.col("tie_firstOrder")
).withColumn(
    "geoDistance_x_secondOrder",
    F.col("geoDistance_z") * F.col("tie_secondOrder")
).withColumn(
    "geoDistance_x_self",
    F.col("geoDistance_z") * F.col("tie_self")
)

interaction_cols = [
    "geoDistance_x_firstOrder",
    "geoDistance_x_secondOrder",
    "geoDistance_x_self"
]

# ---------------------------------------------------------------------------------
# 7. Helper: build polynomial geoDistance terms first, then interaction terms
# ---------------------------------------------------------------------------------
def build_poly_interaction_df(df, degree):
    out = df
    for d in range(1, degree + 1):
        out = out.withColumn(f"geo_z_pow{d}", F.pow(F.col("geoDistance_z"), F.lit(d)))

    for d in range(1, degree + 1):
        out = out.withColumn(f"geo_pow{d}_x_firstOrder", F.col(f"geo_z_pow{d}") * F.col("tie_firstOrder"))
        out = out.withColumn(f"geo_pow{d}_x_secondOrder", F.col(f"geo_z_pow{d}") * F.col("tie_secondOrder"))
        out = out.withColumn(f"geo_pow{d}_x_self",        F.col(f"geo_z_pow{d}") * F.col("tie_self"))
    return out

def get_poly_feature_cols(degree):
    geo_poly_cols = [f"geo_z_pow{d}" for d in range(1, degree + 1)]
    inter_cols = []
    for d in range(1, degree + 1):
        inter_cols += [
            f"geo_pow{d}_x_firstOrder",
            f"geo_pow{d}_x_secondOrder",
            f"geo_pow{d}_x_self",
        ]
    return geo_poly_cols, inter_cols

# polynomial-interaction datasets
df_poly2 = build_poly_interaction_df(df_norm, degree=2)
df_poly3 = build_poly_interaction_df(df_norm, degree=3)

poly2_geo_cols, poly2_interaction_cols = get_poly_feature_cols(degree=2)
poly3_geo_cols, poly3_interaction_cols = get_poly_feature_cols(degree=3)

main_feature_cols = tie_dummy_cols + [f"{c}_z" for c in base_features_continuous]

print("\n=== Descriptive stats: counts and mean(sim) by tie_type ===")
(
    df_norm.groupBy("tie_type")
    .agg(
        F.count("*").alias("n_pairs"),
        F.mean("sim_z").alias("mean_sim"),
        F.stddev("sim_z").alias("sd_sim"),
        F.mean("geoDistance_z").alias("mean_geoDistance"),
    )
    .orderBy("tie_type")
    .show(truncate=False)
)

print("\n=== Overall summary for continuous covariates ===")
(
    df_norm.select(main_feature_cols + ["sim_z"])
    .summary()
    .show(truncate=False)
)

# ---------------------------------------------------------------------------------
# 8. Define feature sets
# ---------------------------------------------------------------------------------
geo_main = "geoDistance_z"
control_continuous = [f"{c}_z" for c in base_features_continuous if c != "geoDistance"]

model_specs = {
    "tie_plus_distance": {
        "df": df_norm,
        "features": tie_dummy_cols + [geo_main],
    },
    "tie_plus_distance_plus_controls": {
        "df": df_norm,
        "features": tie_dummy_cols + [geo_main] + control_continuous,
    },
    "tie_plus_distance_controls_interaction": {
        "df": df_norm,
        "features": tie_dummy_cols + [geo_main] + control_continuous + interaction_cols,
    },
    "tie_plus_distance_controls_poly2_interaction": {
        "df": df_poly2,
        "features": tie_dummy_cols + poly2_geo_cols + control_continuous + poly2_interaction_cols,
    },
    "tie_plus_distance_controls_poly3_interaction": {
        "df": df_poly3,
        "features": tie_dummy_cols + poly3_geo_cols + control_continuous + poly3_interaction_cols,
    },
}

# ---------------------------------------------------------------------------------
# 9. Helper: fit regression + extract table
# ---------------------------------------------------------------------------------
def fit_regression_model(df, feature_cols, label_col="sim_z", model_name="model"):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_vec",
    )
    df_model = assembler.transform(df).select(label_col, "features_vec")

    lr = LinearRegression(
        featuresCol="features_vec",
        labelCol=label_col,
        predictionCol="sim_pred",
        fitIntercept=True,
        regParam=0.0,
        elasticNetParam=0.0,
        solver="normal",
    )
    lr_model = lr.fit(df_model)
    summary = lr_model.summary

    coef_names = ["(Intercept)"] + feature_cols
    coefs = [lr_model.intercept] + list(lr_model.coefficients)

    t_vals = summary.tValues
    p_vals = summary.pValues

    se_coefs = list(summary.coefficientStandardErrors)
    coef_var = [se**2 if se is not None else None for se in se_coefs]

    reg_table = {
        "names": coef_names,
        "coef": coefs,
        "variance": coef_var,
        "t_values": t_vals,
        "p_values": p_vals,
        "r2": summary.r2,
        "r2_adj": summary.r2adj,
        "n_obs": summary.numInstances,
        "model_name": model_name,
    }

    return lr_model, summary, reg_table

# ---------------------------------------------------------------------------------
# 10. Fit models and save to JSON
# ---------------------------------------------------------------------------------
regression_table = {}
model_objects = {}

for mname, spec in model_specs.items():
    feats = spec["features"]
    df_model = spec["df"]

    print(f"\n=== Fitting model: {mname} ===")
    lr_model, summary, reg_table = fit_regression_model(
        df_model,
        feature_cols=feats,
        label_col="sim_z",
        model_name=mname,
    )
    model_objects[mname] = lr_model
    regression_table[mname] = reg_table

    print(f"  R^2={summary.r2:.4f}, Adj R^2={summary.r2adj:.4f}")
    for name, coef, tval, pval in zip(
        reg_table["names"],
        reg_table["coef"],
        reg_table["t_values"],
        reg_table["p_values"],
    ):
        print(f"  {name:40s} coef={coef: .5f} t-value={tval: .5f} p-value={pval: .5f}")

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

serializable = make_json_serializable(regression_table)
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved regression models to JSON: {output_json_path}")

# ---------------------------------------------------------------------------------
# 11. Visualization 1: Two MAIN TERMS in ONE figure, 5 model panels
# ---------------------------------------------------------------------------------
results_dir = os.path.dirname(output_json_path)
with open(output_json_path, "r") as f:
    reg_table_all = json.load(f)

panel_order = [
    ("tie_plus_distance", "A. Model 1: Tie + Distance"),
    ("tie_plus_distance_plus_controls", "B. Model 2: + Controls")
]

keep_terms = ["tie_self", "tie_firstOrder", "tie_secondOrder", "geoDistance_z"]
term_labels = {
    "tie_self": "Self",
    "tie_firstOrder": "First-Order Tie",
    "tie_secondOrder": "Second-Order Tie",
    "geoDistance_z": "Geographical Distance",
}

CI_MULT = 20

def extract_terms_for_model(m, keep_terms):
    names = m["names"]
    coefs = m["coef"]
    variances = m["variance"]
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    rows = []
    for t in keep_terms:
        if t not in name_to_idx:
            rows.append((t, np.nan, np.nan, np.nan, np.nan))
            continue
        i = name_to_idx[t]
        coef = float(coefs[i])
        var = variances[i]
        se = math.sqrt(var) if var is not None and var >= 0 else np.nan
        ci_low = coef - CI_MULT * se
        ci_high = coef + CI_MULT * se
        rows.append((t, coef, se, ci_low, ci_high))
    return rows

terms_order = [term_labels[t] for t in keep_terms]
y_positions = np.arange(len(terms_order))

fig, axes = plt.subplots(
    nrows=1,
    ncols=len(panel_order),
    figsize=(12, 4.8),
    sharey=True
)

if len(panel_order) == 1:
    axes = [axes]

for ax, (mname, mlabel) in zip(axes, panel_order):
    m = reg_table_all[mname]
    rows = extract_terms_for_model(m, keep_terms)

    coefs = np.array([r[1] for r in rows], dtype=float)
    ci_low = np.array([r[3] for r in rows], dtype=float)
    ci_high = np.array([r[4] for r in rows], dtype=float)

    ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.6, color="black")
    ax.errorbar(
        coefs,
        y_positions,
        xerr=[coefs - ci_low, ci_high - coefs],
        fmt="o",
        capsize=4,
        alpha=0.9,
        color="black"
    )
    ax.set_title(mlabel, fontsize=12)
    ax.set_xlabel(f"Coefficient (CI = ± {int(CI_MULT)}×SE)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels(terms_order)
axes[0].set_ylabel("")

plt.tight_layout()
coef4_path = os.path.join(results_dir, "coef_2terms_panels_CI_20xSE.png")
plt.savefig(coef4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {coef4_path}")

# ---------------------------------------------------------------------------------
# 12. Smoothed interaction plot via polynomial geoDistance terms
#     + CI ribbon for fitted line: ± 2×SE(pred), diagonal-only approximation
# ---------------------------------------------------------------------------------
CI_MULT_LINE = 2

def fit_poly_interaction_model_with_var(df, degree=3):
    controls = [f"{c}_z" for c in base_features_continuous if c != "geoDistance"]
    geo_pows = [f"geo_z_pow{d}" for d in range(1, degree + 1)]
    inters = []
    for d in range(1, degree + 1):
        inters += [
            f"geo_pow{d}_x_self",
            f"geo_pow{d}_x_firstOrder",
            f"geo_pow{d}_x_secondOrder",
        ]
    feats = tie_dummy_cols + controls + geo_pows + inters

    assembler = VectorAssembler(inputCols=feats, outputCol="features_vec")
    df_model = assembler.transform(df).select("sim_z", "features_vec")

    lr = LinearRegression(
        featuresCol="features_vec",
        labelCol="sim_z",
        predictionCol="sim_pred",
        fitIntercept=True,
        regParam=0.0,
        elasticNetParam=0.0,
        solver="normal",
    )
    model = lr.fit(df_model)
    summary = model.summary

    se_all = list(summary.coefficientStandardErrors)
    var_all = [(se**2 if se is not None else np.nan) for se in se_all]

    intercept = float(model.intercept)
    coef = np.array(model.coefficients, dtype=float)

    var_intercept = float(var_all[0]) if len(var_all) > 0 and np.isfinite(var_all[0]) else np.nan
    var_feats = var_all[1:1 + len(feats)]
    if len(var_feats) < len(feats):
        var_feats = list(var_feats) + [np.nan] * (len(feats) - len(var_feats))
    var_feats = np.array([v if np.isfinite(v) else np.nan for v in var_feats], dtype=float)

    return feats, intercept, coef, var_intercept, var_feats

def build_design_matrix_poly(x, tie_kind, feats, degree):
    tie_vals = {
        "tie_self": 1.0 if tie_kind == "self" else 0.0,
        "tie_firstOrder": 1.0 if tie_kind == "firstOrder" else 0.0,
        "tie_secondOrder": 1.0 if tie_kind == "secondOrder" else 0.0,
    }

    X = np.zeros((len(x), len(feats)), dtype=float)
    feat_idx = {f: i for i, f in enumerate(feats)}

    for tname, tval in tie_vals.items():
        if tname in feat_idx:
            X[:, feat_idx[tname]] = tval

    for d in range(1, degree + 1):
        col = x ** d

        nm = f"geo_z_pow{d}"
        if nm in feat_idx:
            X[:, feat_idx[nm]] = col

        nm_self = f"geo_pow{d}_x_self"
        nm_fo   = f"geo_pow{d}_x_firstOrder"
        nm_so   = f"geo_pow{d}_x_secondOrder"

        if nm_self in feat_idx:
            X[:, feat_idx[nm_self]] = col * tie_vals["tie_self"]
        if nm_fo in feat_idx:
            X[:, feat_idx[nm_fo]] = col * tie_vals["tie_firstOrder"]
        if nm_so in feat_idx:
            X[:, feat_idx[nm_so]] = col * tie_vals["tie_secondOrder"]

    return X

def predict_curve_and_ci(x, tie_kind, feats, intercept, coef, var_intercept, var_feats, degree, ci_mult=20.0):
    X = build_design_matrix_poly(x, tie_kind, feats, degree)
    y = intercept + X.dot(coef)

    vi = 0.0 if not np.isfinite(var_intercept) else float(var_intercept)
    vf = np.array(var_feats, dtype=float)
    vf[~np.isfinite(vf)] = 0.0

    var_pred = vi + (X**2).dot(vf)
    se_pred = np.sqrt(np.maximum(var_pred, 0.0))

    lo = y - ci_mult * se_pred
    hi = y + ci_mult * se_pred
    return y, lo, hi

x = np.linspace(-2.5, 2.5, 250)

last_deg = None
last_feats = None
last_b0 = None
last_b = None
last_var_b0 = None
last_var_b = None

for deg in [1, 2, 3]:
    print(f"\n=== Fitting polynomial interaction model (degree={deg}) ===")
    df_poly = build_poly_interaction_df(df_norm, degree=deg)
    feats, b0, b, var_b0, var_b = fit_poly_interaction_model_with_var(df_poly, degree=deg)

    print(f"Intercept: {b0: .6f}")
    for fn, cc in zip(feats, b):
        print(f"{fn:35s} {cc: .6f}")

    y_env,  lo_env,  hi_env  = predict_curve_and_ci(x, "environment", feats, b0, b, var_b0, var_b, deg, ci_mult=CI_MULT_LINE)
    y_self, lo_self, hi_self = predict_curve_and_ci(x, "self",        feats, b0, b, var_b0, var_b, deg, ci_mult=CI_MULT_LINE)
    y_fo,   lo_fo,   hi_fo   = predict_curve_and_ci(x, "firstOrder",  feats, b0, b, var_b0, var_b, deg, ci_mult=CI_MULT_LINE)
    y_so,   lo_so,   hi_so   = predict_curve_and_ci(x, "secondOrder", feats, b0, b, var_b0, var_b, deg, ci_mult=CI_MULT_LINE)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.5)

    ax.plot(x, y_env,  linewidth=2, label="Environment")
    ax.fill_between(x, lo_env, hi_env, alpha=0.10)

    ax.plot(x, y_self, linewidth=2, label="Self")
    ax.fill_between(x, lo_self, hi_self, alpha=0.10)

    ax.plot(x, y_fo,   linewidth=2, label="First-Order Tie")
    ax.fill_between(x, lo_fo, hi_fo, alpha=0.10)

    ax.plot(x, y_so,   linewidth=2, label="Second-Order Tie")
    ax.fill_between(x, lo_so, hi_so, alpha=0.10)

    ax.set_xlabel("Geographical Distance")
    ax.set_ylabel("Predicted Learning Rate")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    outp = os.path.join(results_dir, f"interaction_poly_degree_{deg}_CI_{int(CI_MULT_LINE)}xSE.png")
    plt.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outp}")

    last_deg, last_feats, last_b0, last_b = deg, feats, b0, b
    last_var_b0, last_var_b = var_b0, var_b

# ---------------------------------------------------------------------------------
# 13. FINAL: Interaction plot (degree=3 by default) + REAL DATA POINTS (binned means)
# ---------------------------------------------------------------------------------
overlay_deg = last_deg
overlay_feats = last_feats
overlay_b0 = last_b0
overlay_b = last_b
overlay_var_b0 = last_var_b0
overlay_var_b = last_var_b

y_env,  lo_env,  hi_env  = predict_curve_and_ci(x, "environment", overlay_feats, overlay_b0, overlay_b, overlay_var_b0, overlay_var_b, overlay_deg, ci_mult=CI_MULT_LINE)
y_self, lo_self, hi_self = predict_curve_and_ci(x, "self",        overlay_feats, overlay_b0, overlay_b, overlay_var_b0, overlay_var_b, overlay_deg, ci_mult=CI_MULT_LINE)
y_fo,   lo_fo,   hi_fo   = predict_curve_and_ci(x, "firstOrder",  overlay_feats, overlay_b0, overlay_b, overlay_var_b0, overlay_var_b, overlay_deg, ci_mult=CI_MULT_LINE)
y_so,   lo_so,   hi_so   = predict_curve_and_ci(x, "secondOrder", overlay_feats, overlay_b0, overlay_b, overlay_var_b0, overlay_var_b, overlay_deg, ci_mult=CI_MULT_LINE)

bounds = df_norm.select(
    F.min("geoDistance_z").alias("xmin"),
    F.max("geoDistance_z").alias("xmax"),
).collect()[0]
xmin = float(bounds["xmin"])
xmax = float(bounds["xmax"])

n_bins = 60
bin_w = (xmax - xmin) / n_bins if xmax > xmin else 1.0

df_bins = (
    df_norm
    .select("tie_type", "geoDistance_z", "sim_z")
    .withColumn(
        "bin",
        F.when(F.lit(bin_w) > 0,
               F.floor((F.col("geoDistance_z") - F.lit(xmin)) / F.lit(bin_w)).cast("int"))
         .otherwise(F.lit(0))
    )
    .withColumn("bin", F.when(F.col("bin") < 0, 0).otherwise(F.col("bin")))
    .withColumn("bin", F.when(F.col("bin") >= n_bins, n_bins - 1).otherwise(F.col("bin")))
)

bin_means = (
    df_bins
    .groupBy("tie_type", "bin")
    .agg(
        F.mean("geoDistance_z").alias("x_mean"),
        F.mean("sim_z").alias("y_mean"),
        F.count("*").alias("n"),
    )
    .orderBy("tie_type", "bin")
)

bin_pd = bin_means.toPandas()

fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.5)

ax.plot(x, y_env, linewidth=2, label="Environment (fit)")
ax.fill_between(x, lo_env, hi_env, alpha=0.10)

ax.plot(x, y_self, linewidth=2, label="Self (fit)")
ax.fill_between(x, lo_self, hi_self, alpha=0.10)

ax.plot(x, y_fo, linewidth=2, label="First-Order Tie (fit)")
ax.fill_between(x, lo_fo, hi_fo, alpha=0.10)

ax.plot(x, y_so, linewidth=2, label="Second-Order Tie (fit)")
ax.fill_between(x, lo_so, hi_so, alpha=0.10)

label_map = {
    "environment": "Environment (mean)",
    "self": "Self (mean)",
    "firstOrder": "First-Order Tie (mean)",
    "secondOrder": "Second-Order Tie (mean)",
}
for tt, sub in bin_pd.groupby("tie_type"):
    ax.scatter(
        sub["x_mean"].values,
        sub["y_mean"].values,
        s=18,
        alpha=0.55,
        label=label_map.get(tt, f"{tt} (mean)")
    )

ax.set_xlabel("Geographical Distance")
ax.set_ylabel("Predicted Learning Rate")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(ncol=2, fontsize=9)

plt.tight_layout()
overlay_path = os.path.join(results_dir, f"interaction_poly_degree_{overlay_deg}_with_binned_means_CI_{int(CI_MULT_LINE)}xSE.png")
plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {overlay_path}")

spark.stop()