"""No-show prediction for a Digital HealthHub (leakage-safe, patient-grouped validation).

How to run:
    pip install pandas numpy scikit-learn xgboost matplotlib
    python experiment.py KaggleV2-May-2016.csv results

Dataset: Medical Appointment No-Shows (J. Hoppen, Kaggle).

The output folder gets results.json and five figures that match the paper:
    fig1_no_show_rate.png       Fig. 1  no-show rate by lead time and age group
    fig2_roc_pr.png             Fig. 2  ROC and precision-recall curves (test set)
    fig3_confusion_matrix.png   Fig. 3  confusion matrix (XGB, test set)
    fig4_feature_importance.png Fig. 4  permutation importance (XGB)
    fig5_calibration_gains.png  Fig. 5  calibration and cumulative gains (XGB)

Model names used everywhere: LR, DT, RF, HGB, XGB.
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, balanced_accuracy_score,
                             brier_score_loss, roc_curve, precision_recall_curve,
                             confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42

if len(sys.argv) != 3:
    sys.exit("Usage: python experiment.py <input_csv> <output_dir>")
inp, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- load and clean
df = pd.read_csv(inp)
n_raw = len(df)
df = df.rename(columns={
    "Hipertension": "hypertension", "Hypertension": "hypertension",
    "Handcap": "disability", "Handicap": "disability",
    "No-show": "no_show", "Scholarship": "welfare_enrolment",
    "SMS_received": "sms_reminder", "Diabetes": "diabetes",
    "Alcoholism": "alcoholism", "Age": "age"})
df["y"] = (df["no_show"] == "Yes").astype(int)   # y = 1: the patient is a no-show
df["booking_time"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
df["appointment_date"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None).dt.normalize()
df["lead_time"] = (df["appointment_date"] - df["booking_time"].dt.normalize()).dt.days

bad_age = (df.age < 0) | (df.age > 100)
bad_lead_time = df.lead_time < 0    # appointment date before booking date
n_removed = int((bad_age | bad_lead_time).sum())
df = df[~(bad_age | bad_lead_time)].copy()
df["disability"] = df["disability"].clip(upper=1)   # yes/no (levels 2-4 are very rare)
df["PatientId"] = df["PatientId"].astype("int64")

# ---------------------------------------------------------------- features
df["female"] = (df.Gender == "F").astype(int)
df["booking_weekday"] = df.booking_time.dt.dayofweek
df["appointment_weekday"] = df.appointment_date.dt.dayofweek
df["booking_hour"] = df.booking_time.dt.hour
df["same_day_booking"] = (df.lead_time == 0).astype(int)
df["log_lead_time"] = np.log1p(df.lead_time)

# Patient history is leakage-safe: it uses only appointments on strictly earlier dates.
df = df.sort_values(["PatientId", "appointment_date", "booking_time"]).reset_index(drop=True)
g = df.groupby("PatientId")
rows_before_same_date = df.groupby(["PatientId", "appointment_date"]).cumcount()  # same patient, same date
df["prior_appointments"] = g.cumcount().values - rows_before_same_date.values
prior_no_shows_all = g["y"].cumsum().values - df["y"].values
prior_no_shows_same_date = (df.groupby(["PatientId", "appointment_date"])["y"].cumsum().values
                            - df["y"].values)
df["prior_no_shows"] = prior_no_shows_all - prior_no_shows_same_date
df["prior_no_show_rate"] = np.where(df.prior_appointments > 0,
                                    df.prior_no_shows / df.prior_appointments.replace(0, np.nan), np.nan)
df["prior_no_show_rate"] = df["prior_no_show_rate"].fillna(-1.0)   # -1 = no history (new patient)
df["has_history"] = (df.prior_appointments > 0).astype(int)
# Measured from the patient's previous appointment, so it is 0 if that appointment is on the same date.
# Only the date is used here, never the outcome.
df["days_since_previous_appointment"] = (
    df.appointment_date - g["appointment_date"].shift(1)).dt.days.fillna(-1)

# Neighbourhood frequency: counted on the full dataset before the split. It does not use the target.
df["neighbourhood_frequency"] = df["Neighbourhood"].map(df["Neighbourhood"].value_counts(normalize=True))

# Feature groups (Table I)
PATIENT_DETAILS = ["age", "female", "welfare_enrolment", "hypertension", "diabetes", "alcoholism",
                   "disability", "sms_reminder"]
BOOKING = ["lead_time", "log_lead_time", "same_day_booking", "booking_weekday",
           "appointment_weekday", "booking_hour", "neighbourhood_frequency"]
PATIENT_HISTORY = ["prior_appointments", "prior_no_shows", "prior_no_show_rate", "has_history",
                   "days_since_previous_appointment"]
FULL = PATIENT_DETAILS + BOOKING + PATIENT_HISTORY
LEAD_TIME_FEATURES = ["lead_time", "log_lead_time", "same_day_booking"]
LABEL = {"age": "Age", "female": "Gender (female)", "welfare_enrolment": "Welfare enrolment",
         "hypertension": "Hypertension", "diabetes": "Diabetes", "alcoholism": "Alcoholism",
         "disability": "Disability", "sms_reminder": "SMS reminder", "lead_time": "Lead time (days)",
         "log_lead_time": "Log lead time", "same_day_booking": "Same-day booking",
         "booking_weekday": "Booking weekday", "appointment_weekday": "Appointment weekday",
         "booking_hour": "Booking hour", "neighbourhood_frequency": "Neighbourhood frequency",
         "prior_appointments": "Prior appointments", "prior_no_shows": "Prior no-shows",
         "prior_no_show_rate": "Prior no-show rate", "has_history": "Has-history flag",
         "days_since_previous_appointment": "Days since previous appointment"}

# ---------------------------------------------------------------- descriptive statistics
desc = {
    "n_raw": n_raw, "n_removed": n_removed, "n": len(df),
    "n_patients": int(df.PatientId.nunique()),
    "no_show_rate": float(df.y.mean()),
    "date_min": str(df.appointment_date.min().date()), "date_max": str(df.appointment_date.max().date()),
    "age_mean": float(df.age.mean()), "age_sd": float(df.age.std()),
    "female_pct": float(df.female.mean() * 100),
    "sms_pct": float(df.sms_reminder.mean() * 100),
    "lead_time_mean": float(df.lead_time.mean()), "lead_time_median": float(df.lead_time.median()),
    "same_day_pct": float(df.same_day_booking.mean() * 100),
    "no_show_by_sms": df.groupby("sms_reminder").y.mean().to_dict(),
    "no_show_same_day": float(df[df.same_day_booking == 1].y.mean()),
    "no_show_not_same_day": float(df[df.same_day_booking == 0].y.mean()),
    "no_show_by_welfare": df.groupby("welfare_enrolment").y.mean().to_dict(),
    "no_show_by_gender_female": df.groupby("female").y.mean().to_dict(),
    "no_show_by_has_history": df.groupby("has_history").y.mean().to_dict(),
}
lead_time_bins = pd.cut(df.lead_time, [-1, 0, 3, 7, 14, 30, 200],
                        labels=["0", "1-3", "4-7", "8-14", "15-30", ">30"])
desc["no_show_by_lead_time"] = df.groupby(lead_time_bins, observed=True).y.mean().round(4).to_dict()
age_bins = pd.cut(df.age, [-1, 17, 34, 49, 64, 101], labels=["0-17", "18-34", "35-49", "50-64", "65+"])
desc["no_show_by_age"] = df.groupby(age_bins, observed=True).y.mean().round(4).to_dict()
desc["n_by_lead_time"] = df.groupby(lead_time_bins, observed=True).size().to_dict()
desc["n_by_age"] = df.groupby(age_bins, observed=True).size().to_dict()
# SMS reminders are only sent when the lead time is long enough (confounding, see Section VIII-B)
desc["no_show_by_sms_lead_time_gt3"] = df[df.lead_time > 3].groupby("sms_reminder").y.mean().to_dict()
desc["sms_pct_lead_time_le2"] = float(df[df.lead_time <= 2].sms_reminder.mean() * 100)
desc["sms_pct_lead_time_gt3"] = float(df[df.lead_time > 3].sms_reminder.mean() * 100)

# ---------------------------------------------------------------- patient-grouped train / test split
X_all, y_all, groups = df[FULL], df["y"].values, df["PatientId"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X_all, y_all, groups))
desc["n_train"], desc["n_test"] = len(train_idx), len(test_idx)
desc["train_no_show_rate"] = float(y_all[train_idx].mean())
desc["test_no_show_rate"] = float(y_all[test_idx].mean())
assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, "a patient is in both train and test"
pos_weight = (1 - y_all[train_idx].mean()) / y_all[train_idx].mean()   # about 3.94


def make_xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                         scale_pos_weight=pos_weight, eval_metric="auc", n_jobs=2,
                         random_state=SEED, tree_method="hist")


def make_models():
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)),
        "DT": DecisionTreeClassifier(max_depth=6, min_samples_leaf=100,
                                     class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=30,
                                     class_weight="balanced_subsample", n_jobs=2,
                                     random_state=SEED),
        "HGB": HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=300,
                                              min_samples_leaf=60, l2_regularization=1.0,
                                              class_weight="balanced", random_state=SEED),
        "XGB": make_xgb(),
    }


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.nanargmax(f1[:-1])])


def metrics(y, p, threshold):
    y_pred = (p >= threshold).astype(int)
    return {"AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "F1": f1_score(y, y_pred), "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred), "BalAcc": balanced_accuracy_score(y, y_pred),
            "Brier": brier_score_loss(y, p), "threshold": threshold}


def bootstrap_auc_ci(y, p, n=300, seed=SEED):
    """95% percentile bootstrap interval for AUC (resamples records, not patients)."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------- five-fold patient-grouped cross-validation (training set)
X_train, y_train, groups_train = X_all.iloc[train_idx], y_all[train_idx], groups[train_idx]
group_kfold = GroupKFold(n_splits=5)
cv_results, out_of_fold = {}, {}
for name in make_models():
    fold_aucs, fold_pr_aucs = [], []
    oof = np.zeros(len(y_train))
    for fit_idx, val_idx in group_kfold.split(X_train, y_train, groups_train):
        model = make_models()[name]
        model.fit(X_train.iloc[fit_idx], y_train[fit_idx])
        p = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        fold_aucs.append(roc_auc_score(y_train[val_idx], p))
        fold_pr_aucs.append(average_precision_score(y_train[val_idx], p))
    out_of_fold[name] = oof
    cv_results[name] = {"AUC_mean": float(np.mean(fold_aucs)), "AUC_sd": float(np.std(fold_aucs)),
                        "PR-AUC_mean": float(np.mean(fold_pr_aucs)),
                        "PR-AUC_sd": float(np.std(fold_pr_aucs))}
    print("CV", name, cv_results[name], flush=True)

# ---------------------------------------------------------------- test set
X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
test_results, test_scores, fitted = {}, {}, {}
for name in make_models():
    model = make_models()[name]
    model.fit(X_train, y_train)
    fitted[name] = model
    p = model.predict_proba(X_test)[:, 1]
    test_scores[name] = p
    threshold = best_f1_threshold(y_train, out_of_fold[name])   # chosen on out-of-fold training predictions only
    result = metrics(y_test, p, threshold)
    result["AUC_lo"], result["AUC_hi"] = bootstrap_auc_ci(y_test, p)
    test_results[name] = result
    print("TEST", name, {k: round(v, 4) for k, v in result.items()}, flush=True)

best_model = max(test_results, key=lambda k: cv_results[k]["AUC_mean"])   # chosen by CV AUC, not test AUC
print("best model by CV AUC:", best_model)

# ---------------------------------------------------------------- feature-group ablation (XGB), Table III
ablation = {}
feature_sets = {
    "Patient details only": PATIENT_DETAILS,
    "+ Booking and scheduling": PATIENT_DETAILS + BOOKING,
    "+ Patient history (full model)": FULL,
    "Patient history only": PATIENT_HISTORY,
    "Full model without SMS": [f for f in FULL if f != "sms_reminder"],
    "Full model without lead-time features": [f for f in FULL if f not in LEAD_TIME_FEATURES],
}
for set_name, cols in feature_sets.items():
    model = make_xgb()
    model.fit(df.iloc[train_idx][cols], y_train)
    p = model.predict_proba(df.iloc[test_idx][cols])[:, 1]
    lo, hi = bootstrap_auc_ci(y_test, p)
    ablation[set_name] = {"AUC": roc_auc_score(y_test, p), "lo": lo, "hi": hi,
                          "PR-AUC": average_precision_score(y_test, p), "n_features": len(cols)}
    print("ABLATION", set_name, ablation[set_name], flush=True)

# ---------------------------------------------------------------- permutation importance (best model, test set)
perm = permutation_importance(fitted[best_model], X_test, y_test, scoring="roc_auc", n_repeats=5,
                              random_state=SEED, n_jobs=2)
importance = pd.DataFrame({"feature": FULL, "mean": perm.importances_mean, "sd": perm.importances_std}) \
               .sort_values("mean", ascending=False)

# ---------------------------------------------------------------- practical value: lift and capture rate, Table IV
best_scores = test_scores[best_model]
risk_order = np.argsort(-best_scores)
base_rate = y_test.mean()
gains = {}
for fraction in (0.05, 0.10, 0.20, 0.30):
    k = int(len(risk_order) * fraction)
    flagged = y_test[risk_order[:k]]
    gains[f"top_{int(fraction * 100)}%"] = {
        "no_show_rate_in_flagged": float(flagged.mean()),
        "lift": float(flagged.mean() / base_rate),
        "captured": float(flagged.sum() / y_test.sum())}
best_threshold = test_results[best_model]["threshold"]
conf_matrix = confusion_matrix(y_test, (best_scores >= best_threshold).astype(int))

# ---------------------------------------------------------------- extra numbers used to check the paper text
full_auc = ablation["+ Patient history (full model)"]["AUC"]
checks = {
    "AUC_drop_without_lead_time_features": full_auc - ablation["Full model without lead-time features"]["AUC"],
    "AUC_drop_without_sms": full_auc - ablation["Full model without SMS"]["AUC"],
    "AUC_gain_from_patient_history": full_auc - ablation["+ Booking and scheduling"]["AUC"],
    "sms_pct_by_lead_time_days_(5 = 5 or more)": {
        str(k): round(v, 1) for k, v in
        (df.groupby(df.lead_time.clip(upper=5)).sms_reminder.mean() * 100).items()},
}
for label, condition in {"lead_time_gt2": df.lead_time > 2, "lead_time_gt3": df.lead_time > 3,
                         "lead_time_ge3": df.lead_time >= 3}.items():
    subset = df[condition]
    checks[f"no_show_rate_by_sms_{label}"] = {
        str(k): round(v, 4) for k, v in subset.groupby("sms_reminder").y.mean().items()}
    checks[f"sms_pct_{label}"] = float(subset.sms_reminder.mean() * 100)
print("CHECKS", json.dumps(checks, indent=2, default=float))

# ---------------------------------------------------------------- figures (IEEE column width, about 3.4 in)
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300})
COLOUR = {"LR": "#4D4D4D", "DT": "#E69F00", "RF": "#009E73", "HGB": "#0072B2", "XGB": "#D55E00"}

# Fig. 1: no-show rate by lead time and by age group
fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.3))
by_lead_time = pd.Series(desc["no_show_by_lead_time"]) * 100
ax[0].bar(by_lead_time.index.astype(str), by_lead_time.values, color="#0072B2", width=0.65)
for i, v in enumerate(by_lead_time.values):
    ax[0].text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=6)
ax[0].set_xlabel("(a) Lead time (days between booking and appointment)")
ax[0].set_ylabel("No-show rate (%)")
ax[0].set_ylim(0, 40)
by_age = pd.Series(desc["no_show_by_age"]) * 100
ax[1].bar(by_age.index.astype(str), by_age.values, color="#009E73", width=0.65)
for i, v in enumerate(by_age.values):
    ax[1].text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=6)
ax[1].set_xlabel("(b) Patient age group (years)")
ax[1].set_ylabel("No-show rate (%)")
ax[1].set_ylim(0, 30)
plt.tight_layout(pad=0.5, h_pad=1.0)
plt.savefig(f"{out}/fig1_no_show_rate.png")
plt.close()

# Fig. 2: ROC and precision-recall curves (test set)
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.9))
for name, p in test_scores.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, color=COLOUR[name], lw=1.0, label=f"{name} ({test_results[name]['AUC']:.3f})")
    precision, recall, _ = precision_recall_curve(y_test, p)
    ax[1].plot(recall, precision, color=COLOUR[name], lw=1.0)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
ax[0].set_xlabel("False positive rate\n(a) ROC")
ax[0].set_ylabel("True positive rate")
ax[0].legend(fontsize=5, loc="lower right", frameon=False, handlelength=1.2)
ax[1].axhline(base_rate, color="k", ls="--", lw=0.6)
ax[1].set_xlabel("Recall\n(b) Precision-recall")
ax[1].set_ylabel("Precision")
plt.tight_layout(pad=0.5, w_pad=1.0)
plt.savefig(f"{out}/fig2_roc_pr.png")
plt.close()

# Fig. 3: confusion matrix (best model, test set)
fig, ax = plt.subplots(figsize=(2.2, 2.0))
ax.imshow(conf_matrix, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{conf_matrix[i, j]:,}", ha="center", va="center", fontsize=7,
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Attend", "No-show"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Attend", "No-show"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout(pad=0.5)
plt.savefig(f"{out}/fig3_confusion_matrix.png")
plt.close()

# Fig. 4: permutation importance (best model, test set)
top_features = importance.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.barh([LABEL[f] for f in top_features.feature], top_features["mean"], xerr=top_features["sd"],
        color="#0072B2", ecolor="#444", capsize=1.5, height=0.65)
ax.set_xlabel("Decrease in test AUC when feature is permuted")
plt.tight_layout(pad=0.5)
plt.savefig(f"{out}/fig4_feature_importance.png")
plt.close()

# Fig. 5: calibration curve and cumulative gains (best model, test set)
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.9))
observed, predicted = calibration_curve(y_test, best_scores, n_bins=10, strategy="quantile")
ax[0].plot(predicted, observed, "o-", color="#D55E00", ms=2.5, lw=1.0)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
ax[0].set_xlabel("Mean risk score\n(a) Calibration")
ax[0].set_ylabel("Observed no-show rate")
flag_fraction = np.linspace(0, 1, 101)
captured = [y_test[risk_order[:int(len(risk_order) * f)]].sum() / y_test.sum() for f in flag_fraction]
ax[1].plot(flag_fraction * 100, np.array(captured) * 100, color="#D55E00", lw=1.2, label=best_model)
ax[1].plot([0, 100], [0, 100], "k--", lw=0.6, label="Random")
ax[1].set_xlabel("% appointments flagged\n(b) Cumulative gains")
ax[1].set_ylabel("% no-shows captured")
ax[1].legend(frameon=False, fontsize=5.5)
plt.tight_layout(pad=0.5, w_pad=1.0)
plt.savefig(f"{out}/fig5_calibration_gains.png")
plt.close()

# ---------------------------------------------------------------- save all results
results = {"dataset": desc, "cross_validation": cv_results, "test_set": test_results,
           "best_model": best_model, "ablation": ablation,
           "permutation_importance": importance.assign(label=importance.feature.map(LABEL)).to_dict("records"),
           "gains": gains, "confusion_matrix_best_model": conf_matrix.tolist(),
           "features": FULL, "checks": checks}
with open(f"{out}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("DONE")
