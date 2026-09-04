#!/usr/bin/env python3

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def parse_txt_curve(path: Path):
    meta = {}
    x = []
    y = []
    in_extend = False
    columns = None

    with path.open("r", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                s = line[1:].strip()

                if ":" in s:
                    k, v = s.split(":", 1)
                    meta[k.strip()] = v.strip()

                if s.startswith("segment:"):
                    in_extend = (
                        s.split(":", 1)[1].strip().lower()
                        == "extend"
                    )

                elif s.startswith("columns:"):
                    columns = (
                        s.split(":", 1)[1]
                        .strip()
                        .split()
                    )

                continue

            if (
                not in_extend
                or not columns
                or not line.strip()
            ):
                continue

            vals = line.split()

            if len(vals) < len(columns):
                continue

            try:
                xi = float(
                    vals[
                        columns.index(
                            "verticalTipPosition"
                        )
                    ]
                )
                yi = float(
                    vals[
                        columns.index(
                            "vDeflection"
                        )
                    ]
                )
            except Exception:
                continue

            x.append(xi)
            y.append(yi)

    if not x:
        raise ValueError(
            f"No extend curve found in {path}"
        )

    return {
        "path": path,
        "index": int(
            meta.get("index", "-1")
        ),
        "xPosition": float(
            meta.get(
                "xPosition",
                "nan",
            )
        ),
        "yPosition": float(
            meta.get(
                "yPosition",
                "nan",
            )
        ),
        "x": np.asarray(
            x,
            float,
        ),
        "y": np.asarray(
            y,
            float,
        ),
    }


def parse_proc(path: Path):
    out = {}

    if not path or not path.exists():
        return out

    try:
        with zipfile.ZipFile(path) as z:
            name = (
                "header.properties"
                if "header.properties"
                in z.namelist()
                else z.namelist()[0]
            )

            text = z.read(name).decode(
                "utf-8",
                "replace",
            )

    except zipfile.BadZipFile:
        text = path.read_text(
            errors="replace"
        )

    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()

    return out


def get_float(
    d,
    key,
    default=np.nan,
):
    value = d.get(key)

    if value is None:
        return default

    if value == "Infinity":
        return np.inf

    if value == "-Infinity":
        return -np.inf

    try:
        return float(value)
    except Exception:
        return default


def linear_fit_search(
    x,
    y,
    target_slope,
    target_interp=np.nan,
):
    """
    Recover the effective JPK linear-fit X Max.

    For this processing chain:

        fit range = (-Infinity, X Max]

    More-negative verticalTipPosition corresponds
    to the deeper, higher-force part of the curve.
    """

    order = np.argsort(x)

    xs = x[order]
    ys = y[order]

    best = None

    for j in range(
        8,
        len(xs) - 8,
    ):
        if xs[j] > 0:
            break

        xx = xs[: j + 1]
        yy = ys[: j + 1]

        m, b = np.polyfit(
            -xx,
            yy,
            1,
        )

        x0 = (
            b / m
            if m
            else np.nan
        )

        slope_scale = max(
            abs(target_slope) * 1e-5,
            1e-9,
        )

        score = (
            (
                m - target_slope
            )
            / slope_scale
        ) ** 2

        if np.isfinite(
            target_interp
        ):
            score += (
                (
                    x0
                    - target_interp
                )
                / 1e-12
            ) ** 2

        if (
            best is None
            or score
            < best["score"]
        ):
            lo = xs[j]

            hi = (
                xs[j + 1]
                if j + 1
                < len(xs)
                else xs[j]
            )

            best = {
                "xmax_eff": lo,
                "interval_lo": lo,
                "interval_hi": hi,
                "slope": m,
                "intercept": b,
                "x0": x0,
                "err": abs(
                    m - target_slope
                ),
                "score": score,
                "n": len(xx),
            }

    return best


def fit_elasticity_range(
    x,
    y,
    xmin=-np.inf,
    xmax=np.inf,
):
    mask = np.ones_like(
        x,
        dtype=bool,
    )

    if np.isfinite(xmin):
        mask &= x >= xmin

    if np.isfinite(xmax):
        mask &= x <= xmax

    xx = x[mask]
    yy = y[mask]

    if len(xx) < 12:
        return None

    # Scale coordinates for numerical stability.
    xn = xx * 1e9
    yp = yy * 1e12

    span = max(
        np.ptp(xn),
        1e-6,
    )

    lo = float(
        np.min(xn)
        - 0.1 * span
    )

    hi = float(
        min(
            np.max(xn)
            + 0.02 * span,
            50.0,
        )
    )

    def solve_at(xc_nm):
        q = np.maximum(
            xc_nm - xn,
            0.0,
        ) ** 2

        M = np.column_stack(
            [
                np.ones_like(q),
                q,
            ]
        )

        coef, *_ = np.linalg.lstsq(
            M,
            yp,
            rcond=None,
        )

        (
            b_pN,
            A_pN_nm2,
        ) = coef

        if A_pN_nm2 < 0:
            A_pN_nm2 = 0.0

            b_pN = float(
                np.mean(yp)
            )

            pred = np.full_like(
                yp,
                b_pN,
            )

        else:
            pred = (
                b_pN
                + A_pN_nm2 * q
            )

        rss = float(
            np.sum(
                (yp - pred)
                ** 2
            )
        )

        return (
            rss,
            b_pN,
            A_pN_nm2,
            pred,
        )

    res = minimize_scalar(
        lambda xc: solve_at(
            xc
        )[0],
        bounds=(
            lo,
            hi,
        ),
        method="bounded",
        options={
            "xatol": 1e-7
        },
    )

    xc_nm = float(
        res.x
    )

    (
        rss,
        bp,
        Ap,
        pred,
    ) = solve_at(
        xc_nm
    )

    rms_pN = float(
        np.sqrt(
            np.mean(
                (
                    yp
                    - pred
                )
                ** 2
            )
        )
    )

    return {
        "xmin": xmin,
        "xmax": xmax,
        "xc": (
            xc_nm * 1e-9
        ),
        "baseline": (
            float(bp)
            * 1e-12
        ),
        "A": (
            float(Ap)
            * 1e6
        ),
        "rms": (
            rms_pN
            * 1e-12
        ),
        "rss": (
            rss
            * 1e-24
        ),
        "n": len(xx),
    }


def reconstructed_modulus(
    fit,
    proc,
):
    """
    Convert the fitted quadratic Hertz-Sneddon
    coefficient into Young's modulus.

    The current JPK recipe uses a quadratic
    pyramid model.
    """

    if fit is None:
        return np.nan

    A = fit.get(
        "A",
        np.nan,
    )

    nu = get_float(
        proc,
        "operation.5.poisson-ratio",
        0.5,
    )

    angle = get_float(
        proc,
        "operation.5.semi-angle-to-face",
        20.0,
    )

    if (
        not np.isfinite(A)
        or not np.isfinite(nu)
        or not np.isfinite(angle)
    ):
        return np.nan

    tan_angle = math.tan(
        math.radians(
            angle
        )
    )

    if (
        not np.isfinite(
            tan_angle
        )
        or tan_angle == 0
    ):
        return np.nan

    return (
        A
        * (1 - nu**2)
        / (0.7453 * tan_angle)
    )


def elastic_score(
    fit,
    target,
    scales,
    proc,
):
    """
    Compare a reconstructed elasticity fit
    against independent quantities saved by JPK.

    The score includes:

    - contact point
    - baseline
    - ResidualRMS
    - Young's modulus

    Lower score is better.
    """

    if fit is None:
        return np.inf

    terms = []

    for key, tkey in [
        (
            "xc",
            "contact",
        ),
        (
            "baseline",
            "baseline",
        ),
        (
            "rms",
            "rms",
        ),
    ]:
        tv = target.get(
            tkey,
            np.nan,
        )

        if np.isfinite(tv):
            terms.append(
                (
                    (
                        fit[key]
                        - tv
                    )
                    / scales[key]
                )
                ** 2
            )

    target_E = target.get(
        "E",
        np.nan,
    )

    E_rec = (
        reconstructed_modulus(
            fit,
            proc,
        )
    )

    if (
        np.isfinite(
            target_E
        )
        and target_E > 0
        and np.isfinite(
            E_rec
        )
    ):
        # 0.1% relative modulus tolerance.
        # 100 Pa is used as a numerical floor.
        E_scale = max(
            abs(target_E)
            * 0.001,
            100.0,
        )

        terms.append(
            (
                (
                    E_rec
                    - target_E
                )
                / E_scale
            )
            ** 2
        )

    if not terms:
        return np.inf

    return float(
        np.mean(
            terms
        )
    )


def candidate_values(
    x,
    step_nm=0.25,
):
    mn = float(
        np.min(x)
    )

    mx = float(
        np.max(x)
    )

    upper = min(
        0.0,
        mx,
    )

    if upper <= mn:
        upper = mx

    vals = np.arange(
        mn * 1e9
        + step_nm,
        upper * 1e9
        + 1e-9,
        step_nm,
    )

    return (
        vals * 1e-9
    )


def reconstruct_elasticity(
    x,
    y,
    target,
    proc,
    finite_xmax=False,
    step_nm=0.25,
):
    """
    Reconstruct the effective JPK elasticity fit range.

    Search order:
      1. -Infinity -> +Infinity
      2. finite X Min -> +Infinity
      3. finite X Min -> finite X Max, only when needed

    Finite X Max is searched on actual sampled x positions.
    This avoids missing narrow effective JPK boundary intervals.
    """

    scales = {
        "xc": 0.25e-9,
        "baseline": 1e-12,
        "rms": 1e-12,
    }

    tested = []

    def get_score(fit):
        return elastic_score(
            fit,
            target,
            scales,
            proc,
        )

    actual_x = np.sort(
        np.unique(x)
    )

    negative_x = actual_x[
        actual_x < 0
    ]

    positive_x = actual_x[
        actual_x > 0
    ]

    # --------------------------------------------------
    # 1. Full range
    # --------------------------------------------------

    full_fit = fit_elasticity_range(
        x,
        y,
        -np.inf,
        np.inf,
    )

    full_result = (
        "full",
        full_fit,
        get_score(full_fit),
    )

    tested.append(full_result)

    # --------------------------------------------------
    # 2. finite X Min -> +Infinity
    # --------------------------------------------------

    vals = candidate_values(
        x,
        step_nm,
    )

    best1 = None

    for xmin in vals:
        fit = fit_elasticity_range(
            x,
            y,
            xmin,
            np.inf,
        )

        score = get_score(fit)

        if (
            best1 is None
            or score < best1[2]
        ):
            best1 = (
                "xmin_inf",
                fit,
                score,
            )

    # Refine X Min around the coarse optimum using
    # actual sampled x positions.
    if (
        best1 is not None
        and best1[1] is not None
        and len(negative_x) > 0
    ):
        center = best1[1]["xmin"]

        local = negative_x[
            (negative_x >= center - 0.75e-9)
            &
            (negative_x <= center + 0.75e-9)
        ]

        for xmin in local:
            fit = fit_elasticity_range(
                x,
                y,
                xmin,
                np.inf,
            )

            score = get_score(fit)

            if score < best1[2]:
                best1 = (
                    "xmin_inf",
                    fit,
                    score,
                )

    if best1 is not None:
        tested.append(best1)

    simple = min(
        [
            t
            for t in tested
            if t[0] in (
                "full",
                "xmin_inf",
            )
        ],
        key=lambda t: t[2],
    )

    # --------------------------------------------------
    # 3. finite X Min -> finite X Max
    # --------------------------------------------------
    #
    # Do not perform a huge 2D brute-force search.
    #
    # Start from the best finite-XMin solution, then:
    #   A. scan every sampled positive X Max
    #   B. optimize X Min at that X Max
    #   C. rescan every X Max
    #   D. refine X Min once more
    #
    # This is sample-aware and much faster.
    # --------------------------------------------------

    need_2d = (
        finite_xmax
        or simple[2] > 0.01
    )

    best2 = None

    if (
        need_2d
        and best1 is not None
        and best1[1] is not None
        and len(positive_x) > 0
    ):
        current_xmin = best1[1]["xmin"]

        # ----------------------------------------------
        # Pass A: all sampled positive X Max values
        # at the best X Min.
        # ----------------------------------------------

        for xmax in positive_x:
            if xmax <= current_xmin:
                continue

            fit = fit_elasticity_range(
                x,
                y,
                current_xmin,
                xmax,
            )

            score = get_score(fit)

            if (
                best2 is None
                or score < best2[2]
            ):
                best2 = (
                    "finite_both",
                    fit,
                    score,
                )

        # ----------------------------------------------
        # Pass B: optimize X Min around the finite-both
        # candidate while holding X Max fixed.
        # ----------------------------------------------

        if (
            best2 is not None
            and best2[1] is not None
        ):
            current_xmax = best2[1]["xmax"]

            xmin_center = best2[1]["xmin"]

            xmin_local = negative_x[
                (
                    negative_x
                    >= xmin_center - 1.5e-9
                )
                &
                (
                    negative_x
                    <= xmin_center + 1.5e-9
                )
            ]

            for xmin in xmin_local:
                if xmin >= current_xmax:
                    continue

                fit = fit_elasticity_range(
                    x,
                    y,
                    xmin,
                    current_xmax,
                )

                score = get_score(fit)

                if score < best2[2]:
                    best2 = (
                        "finite_both",
                        fit,
                        score,
                    )

        # ----------------------------------------------
        # Pass C: rescan every sampled positive X Max
        # at the refined X Min.
        # ----------------------------------------------

        if (
            best2 is not None
            and best2[1] is not None
        ):
            current_xmin = best2[1]["xmin"]

            for xmax in positive_x:
                if xmax <= current_xmin:
                    continue

                fit = fit_elasticity_range(
                    x,
                    y,
                    current_xmin,
                    xmax,
                )

                score = get_score(fit)

                if score < best2[2]:
                    best2 = (
                        "finite_both",
                        fit,
                        score,
                    )

        # ----------------------------------------------
        # Pass D: final local X Min refinement.
        # ----------------------------------------------

        if (
            best2 is not None
            and best2[1] is not None
        ):
            current_xmax = best2[1]["xmax"]
            xmin_center = best2[1]["xmin"]

            xmin_local = negative_x[
                (
                    negative_x
                    >= xmin_center - 0.5e-9
                )
                &
                (
                    negative_x
                    <= xmin_center + 0.5e-9
                )
            ]

            for xmin in xmin_local:
                if xmin >= current_xmax:
                    continue

                fit = fit_elasticity_range(
                    x,
                    y,
                    xmin,
                    current_xmax,
                )

                score = get_score(fit)

                if score < best2[2]:
                    best2 = (
                        "finite_both",
                        fit,
                        score,
                    )

        if best2 is not None:
            tested.append(best2)

    # --------------------------------------------------
    # Conservative model selection
    # --------------------------------------------------

    best = simple

    if best2 is not None:
        finite_score = best2[2]
        simple_score = simple[2]

        # The extra X Max boundary is accepted only when
        # the numerical evidence is very strong.
        if (
            finite_score < 0.01
            and finite_score
            < simple_score * 0.10
        ):
            best = best2

    return (
        best,
        tested,
    )

def boundary_interval_for_value(
    x,
    value,
    kind,
):
    xs = np.sort(
        np.unique(x)
    )

    if not np.isfinite(
        value
    ):
        return (
            value,
            value,
        )

    if kind == "xmin":
        included = xs[
            xs >= value
        ]

        if not len(
            included
        ):
            return (
                value,
                value,
            )

        first = included[0]

        lower = xs[
            xs < first
        ]

        return (
            (
                lower[-1]
                if len(lower)
                else -np.inf
            ),
            first,
        )

    included = xs[
        xs <= value
    ]

    if not len(
        included
    ):
        return (
            value,
            value,
        )

    last = included[-1]

    upper = xs[
        xs > last
    ]

    return (
        last,
        (
            upper[0]
            if len(upper)
            else np.inf
        ),
    )


def find_dataset_files(
    folder: Path,
):
    tsvs = list(
        folder.glob(
            "*.tsv"
        )
    )

    procs = list(
        folder.glob(
            "*.jpk-proc-force"
        )
    )

    txts = [
        p
        for p
        in folder.glob(
            "*.txt"
        )
        if not p.name.lower().endswith(
            (
                "-young-modulus-extend.txt",
                "-baseline-extend.txt",
                "-slope-extend.txt",
                "-contact-point-extend.txt",
            )
        )
    ]

    return (
        (
            tsvs[0]
            if tsvs
            else None
        ),
        (
            procs[0]
            if procs
            else None
        ),
        txts,
    )


def process_folder(
    folder: Path,
    outdir: Path,
    force_finite_xmax=False,
    step_nm=0.25,
):
    (
        tsv,
        proc_path,
        txts,
    ) = find_dataset_files(
        folder
    )

    if not tsv:
        raise FileNotFoundError(
            f"No TSV found in {folder}"
        )

    if not txts:
        raise FileNotFoundError(
            "No individual curve TXT "
            f"files found in {folder}"
        )

    df = pd.read_csv(
        tsv,
        sep="\t",
    )

    proc = (
        parse_proc(
            proc_path
        )
        if proc_path
        else {}
    )

    rows = []

    by_index = {
        int(
            r[
                "Position Index"
            ]
        ): r
        for _, r
        in df.iterrows()
    }

    for txt in sorted(
        txts
    ):
        try:
            curve = (
                parse_txt_curve(
                    txt
                )
            )

        except Exception as e:
            print(
                f"SKIP {txt.name}: {e}"
            )
            continue

        if (
            curve["index"]
            not in by_index
        ):
            print(
                f"SKIP {txt.name}: "
                f'index {curve["index"]} '
                "not in TSV"
            )
            continue

        r = by_index[
            curve["index"]
        ]

        target_slope = float(
            r.get(
                "Slope [N/m]",
                np.nan,
            )
        )

        target_interp = float(
            r.get(
                "Interpolated Height [m]",
                np.nan,
            )
        )

        if np.isfinite(
            target_slope
        ):
            lin = (
                linear_fit_search(
                    curve["x"],
                    curve["y"],
                    target_slope,
                    target_interp,
                )
            )
        else:
            lin = None

        target = {
            "contact": float(
                r.get(
                    "Contact Point [m]",
                    np.nan,
                )
            ),
            "baseline": float(
                r.get(
                    "Baseline [N]",
                    np.nan,
                )
            ),
            "rms": float(
                r.get(
                    "ResidualRMS [N]",
                    np.nan,
                )
            ),
            "E": float(
                r.get(
                    "Young's Modulus [Pa]",
                    np.nan,
                )
            ),
        }

        (
            best,
            tested,
        ) = (
            reconstruct_elasticity(
                curve["x"],
                curve["y"],
                target,
                proc,
                force_finite_xmax,
                step_nm,
            )
        )

        (
            mode,
            efit,
            score,
        ) = best

        xmin_int = (
            boundary_interval_for_value(
                curve["x"],
                efit["xmin"],
                "xmin",
            )
        )

        xmax_int = (
            boundary_interval_for_value(
                curve["x"],
                efit["xmax"],
                "xmax",
            )
        )

        E_rec = (
            reconstructed_modulus(
                efit,
                proc,
            )
        )

        finite_scores = {
            m: s
            for m, _, s
            in tested
        }

        scores = sorted(
            [
                s
                for _, _, s
                in tested
            ]
        )

        second = (
            scores[1]
            if len(scores) > 1
            else np.inf
        )

        ratio = (
            second
            / max(
                score,
                1e-12,
            )
        )

        if (
            score < 0.25
            and ratio > 3
        ):
            confidence = (
                "high"
            )

        elif (
            score < 1.0
            and ratio > 1.5
        ):
            confidence = (
                "medium"
            )

        else:
            confidence = (
                "low"
            )

        rows.append(
            {
                "file":
                    txt.name,

                "position_index":
                    curve[
                        "index"
                    ],

                "x_position_m":
                    curve[
                        "xPosition"
                    ],

                "y_position_m":
                    curve[
                        "yPosition"
                    ],

                "deepest_x_nm":
                    np.min(
                        curve["x"]
                    )
                    * 1e9,

                "linear_xmax_nm":
                    (
                        lin[
                            "xmax_eff"
                        ]
                        * 1e9
                        if lin
                        else np.nan
                    ),

                "linear_interval_lo_nm":
                    (
                        lin[
                            "interval_lo"
                        ]
                        * 1e9
                        if lin
                        else np.nan
                    ),

                "linear_interval_hi_nm":
                    (
                        lin[
                            "interval_hi"
                        ]
                        * 1e9
                        if lin
                        else np.nan
                    ),

                "jpk_slope_N_per_m":
                    target_slope,

                "reconstructed_slope_N_per_m":
                    (
                        lin[
                            "slope"
                        ]
                        if lin
                        else np.nan
                    ),

                "slope_abs_error":
                    (
                        lin[
                            "err"
                        ]
                        if lin
                        else np.nan
                    ),

                "elastic_mode":
                    mode,

                "elastic_xmin_nm":
                    (
                        efit[
                            "xmin"
                        ]
                        * 1e9
                        if np.isfinite(
                            efit[
                                "xmin"
                            ]
                        )
                        else -np.inf
                    ),

                "elastic_xmin_interval_lo_nm":
                    (
                        xmin_int[0]
                        * 1e9
                        if np.isfinite(
                            xmin_int[0]
                        )
                        else -np.inf
                    ),

                "elastic_xmin_interval_hi_nm":
                    (
                        xmin_int[1]
                        * 1e9
                        if np.isfinite(
                            xmin_int[1]
                        )
                        else -np.inf
                    ),

                "elastic_xmax_nm":
                    (
                        efit[
                            "xmax"
                        ]
                        * 1e9
                        if np.isfinite(
                            efit[
                                "xmax"
                            ]
                        )
                        else np.inf
                    ),

                "elastic_xmax_interval_lo_nm":
                    (
                        xmax_int[0]
                        * 1e9
                        if np.isfinite(
                            xmax_int[0]
                        )
                        else np.inf
                    ),

                "elastic_xmax_interval_hi_nm":
                    (
                        xmax_int[1]
                        * 1e9
                        if np.isfinite(
                            xmax_int[1]
                        )
                        else np.inf
                    ),

                "jpk_young_modulus_Pa":
                    target["E"],

                "reconstructed_young_modulus_Pa":
                    E_rec,

                "young_modulus_abs_error_Pa":
                    (
                        abs(
                            E_rec
                            - target["E"]
                        )
                        if (
                            np.isfinite(
                                E_rec
                            )
                            and np.isfinite(
                                target["E"]
                            )
                        )
                        else np.nan
                    ),

                "young_modulus_rel_error_percent":
                    (
                        abs(
                            E_rec
                            - target["E"]
                        )
                        / abs(
                            target["E"]
                        )
                        * 100
                        if (
                            np.isfinite(
                                E_rec
                            )
                            and np.isfinite(
                                target["E"]
                            )
                            and target["E"]
                            != 0
                        )
                        else np.nan
                    ),

                "jpk_contact_point_nm":
                    (
                        target[
                            "contact"
                        ]
                        * 1e9
                    ),

                "reconstructed_contact_point_nm":
                    (
                        efit["xc"]
                        * 1e9
                    ),

                "jpk_baseline_pN":
                    (
                        target[
                            "baseline"
                        ]
                        * 1e12
                    ),

                "reconstructed_baseline_pN":
                    (
                        efit[
                            "baseline"
                        ]
                        * 1e12
                    ),

                "jpk_rms_pN":
                    (
                        target[
                            "rms"
                        ]
                        * 1e12
                    ),

                "reconstructed_rms_pN":
                    (
                        efit["rms"]
                        * 1e12
                    ),

                "elastic_score":
                    score,

                "confidence":
                    confidence,

                "score_full":
                    finite_scores.get(
                        "full",
                        np.nan,
                    ),

                "score_xmin_inf":
                    finite_scores.get(
                        "xmin_inf",
                        np.nan,
                    ),

                "score_finite_both":
                    finite_scores.get(
                        "finite_both",
                        np.nan,
                    ),
            }
        )

        print(
            f"{txt.name}: "
            f'linear={rows[-1]["linear_xmax_nm"]:.3f} nm | '
            f"elasticity={mode} | "
            f'xmin={rows[-1]["elastic_xmin_nm"]} | '
            f'xmax={rows[-1]["elastic_xmax_nm"]} | '
            f"E={E_rec:.6g} Pa | "
            f"score={score:.3g}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = pd.DataFrame(
        rows
    )

    csv_path = (
        outdir
        / (
            folder.name
            + "_recovered_boundaries.csv"
        )
    )

    out.to_csv(
        csv_path,
        index=False,
    )

    try:
        xlsx_path = (
            outdir
            / (
                folder.name
                + "_recovered_boundaries.xlsx"
            )
        )

        out.to_excel(
            xlsx_path,
            index=False,
        )

    except Exception as e:
        xlsx_path = None

        print(
            "Excel output skipped:",
            e,
        )

    return (
        csv_path,
        xlsx_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recover JPK linear and elasticity "
            "fit boundaries from individual "
            "curve TXT files and processed TSV."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Dataset folder, or parent "
            "folder when using --recursive"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fit-boundary/results"
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Process every subfolder "
            "containing a TSV file"
        ),
    )

    parser.add_argument(
        "--force-finite-xmax-search",
        action="store_true",
        help=(
            "Always run the slower finite "
            "elasticity X Min/X Max search"
        ),
    )

    parser.add_argument(
        "--step-nm",
        type=float,
        default=0.25,
        help=(
            "Coarse elasticity X Min "
            "search step in nm"
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.recursive:
        folders = [
            p
            for p
            in [
                args.input,
                *args.input.rglob(
                    "*"
                ),
            ]
            if (
                p.is_dir()
                and any(
                    p.glob(
                        "*.tsv"
                    )
                )
            )
        ]

    else:
        folders = [
            args.input
        ]

    if not folders:
        raise SystemExit(
            "No dataset folders "
            "containing TSV files found."
        )

    for folder in folders:
        print(
            "\n===",
            folder,
            "===",
        )

        try:
            process_folder(
                folder,
                args.output,
                args.force_finite_xmax_search,
                args.step_nm,
            )

        except Exception as e:
            print(
                "ERROR:",
                folder,
                e,
            )


if __name__ == "__main__":
    main()