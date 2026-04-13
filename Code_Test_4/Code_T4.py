"""
=============================================================================
ANALYSE DE VALIDATION — Test T4
Capteur de pression statique Honeywell ABPDJJT001PDSA3
=============================================================================

Structure du CSV attendu (6 colonnes) :
  - id                : identifiant de la mesure
  - date              : date/heure de l'acquisition
  - time_s            : temps en secondes depuis le début de l'acquisition
  - mesure_cmH20      : valeur brute mesurée par le capteur (cm H₂O)
  - valeur_réel_cmH20 : palier théorique appliqué (cm H₂O)
  - essai             : numéro d'essai (1, 2 ou 3)

Sorties produites :
  Tableau T4 - pour chaque palier (amplitude) :
    • Δt essai 1 (s)          = temps depuis t0 jusqu'à stabilisation
    • Val. essai 1 (cm H₂O)   = valeur moyenne sur la fenêtre de stabilisation
    • (idem essais 2 et 3)
    • Δt moyen (s)            = moyenne des Δt des 3 essais
    • Hauteur moy. (cm H₂O)  = moyenne des valeurs stables des 3 essais
    • Erreur (%)              = |hauteur moy. - palier| / palier × 100
    • Conforme (Oui/Non)      = Oui si Δt moyen ≤ LATENCE_MAX_S

Condition de stabilisation :
  Phase 1 - dans ±TOLERANCE_PCT% du palier théorique :
      Si N_STABLE pts consécutifs vérifient |mesure − palier| ≤ tol
      → Δt = time_s[premier pt stable] − t0
      → Val = moyenne de ces N_STABLE points

  Phase 2 - si le palier n'est jamais atteint (biais capteur) :
      h2_reel = moyenne des N_FIN dernières mesures (valeur stable réelle)
      Même condition sur h2_reel à ±TOLERANCE_PCT%
      → Δt et Val calculés sur h2_reel

Utilisation :
  python analyse_test_T4.py                          # lit "pressure_data.csv"
  python analyse_test_T4.py mon_fichier.csv          # lit le fichier spécifié
  python analyse_test_T4.py mon_fichier.csv resultats.csv  # + export CSV
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd

# PARAMÈTRES DU TEST  (ne pas modifier sans raison)
LATENCE_MAX_S  = 0.100   # s   - seuil de conformité (100 ms)
TOLERANCE_PCT  = 10.0    # %   - fenêtre de stabilisation autour du palier
N_STABLE       = 5       # nb de points consécutifs requis dans la fenêtre
N_FIN          = 30      # nb de dernières mesures pour estimer h2 si non atteint
ESSAIS         = [1, 2, 3]

# 1.  CHARGEMENT ET NETTOYAGE DU CSV
def charger_csv(chemin: str) -> pd.DataFrame:
    if not os.path.exists(chemin):
        sys.exit(f"[ERREUR] Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)
    df.columns = df.columns.str.strip()

    colonnes_requises = {"id", "date", "time_s",
                         "mesure_cmH20", "valeur_réel_cmH20", "essai"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        sys.exit(f"[ERREUR] Colonnes manquantes dans le CSV : {manquantes}")

    for col in ["time_s", "mesure_cmH20", "valeur_réel_cmH20", "essai"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_avant = len(df)
    df = df.dropna(subset=["time_s", "mesure_cmH20", "valeur_réel_cmH20", "essai"])
    if len(df) != n_avant:
        print(f"[AVERTISSEMENT] {n_avant - len(df)} ligne(s) ignorée(s).")

    df = df.sort_values("time_s").reset_index(drop=True)
    return df

# 2.  DÉTECTION DE LA LATENCE ET VALEUR DE STABILISATION
def trouver_latence(sub: pd.DataFrame, palier: float) -> tuple:
    """
    Méthode de calcul
    -----------------
    t0  = time_s[0]  — début de l'acquisition (départ du test)
    h1  = première mesure de l'essai

    Phase 1 — Stabilisation autour du palier théorique :
        tol = TOLERANCE_PCT / 100 × palier
        Si N_STABLE points consécutifs vérifient |mesure − palier| ≤ tol :
            Δt         = time_s[premier pt] − t0
            val_stable = moyenne de ces N_STABLE points

    Phase 2 — Si le palier n'est jamais atteint (biais capteur) :
        h2_reel = moyenne des N_FIN dernières mesures
        tol2    = TOLERANCE_PCT / 100 × h2_reel
        Même recherche sur h2_reel → Δt et val_stable sur la valeur réelle

    Retourne : (delta_t, val_stable)
    """
    vals  = sub["mesure_cmH20"].values
    times = sub["time_s"].values
    t0    = times[0]
    tol   = palier * TOLERANCE_PCT / 100.0

    # Phase 1 - palier théorique
    for i in range(len(vals) - N_STABLE + 1):
        fenetre = vals[i:i + N_STABLE]
        if all(abs(v - palier) <= tol for v in fenetre):
            return (round(float(times[i] - t0), 4),
                    round(float(np.mean(fenetre)), 4))

    # Phase 2 - valeur stable réelle
    h2_reel = float(np.mean(vals[-N_FIN:])) if len(vals) >= N_FIN else float(np.mean(vals))
    tol2    = abs(h2_reel) * TOLERANCE_PCT / 100.0
    for i in range(len(vals) - N_STABLE + 1):
        fenetre = vals[i:i + N_STABLE]
        if all(abs(v - h2_reel) <= tol2 for v in fenetre):
            return (round(float(times[i] - t0), 4),
                    round(float(np.mean(fenetre)), 4))

    # Fallback absolu
    return (round(float(times[-1] - t0), 4), round(h2_reel, 4))


# 3.  TABLEAU DES RÉSULTATS PAR PALIER
def construire_tableau(df: pd.DataFrame) -> pd.DataFrame:
    paliers = sorted(df["valeur_réel_cmH20"].unique())
    rows = []

    for palier in paliers:
        sous = df[df["valeur_réel_cmH20"] == palier]

        deltas     = {}
        val_stable = {}

        for e in ESSAIS:
            sub = sous[sous["essai"] == e].reset_index(drop=True)
            if sub.empty:
                continue
            dt, val       = trouver_latence(sub, palier)
            deltas[e]     = dt
            val_stable[e] = val

        if not deltas:
            continue

        dt_valides  = [v for v in deltas.values()     if v is not None]
        val_valides = [v for v in val_stable.values() if v is not None]

        dt_moyen   = round(float(np.mean(dt_valides)),          4) if dt_valides  else None
        haut_moy   = round(float(np.mean(val_valides)),         4) if val_valides else None
        erreur_pct = round(abs(haut_moy - palier) / palier * 100, 2) \
                     if haut_moy is not None else None
        conforme   = "Oui" if (dt_moyen is not None and dt_moyen <= LATENCE_MAX_S) else "Non"

        row = {"Palier (cm H₂O)": int(palier) if palier == int(palier) else palier}
        for e in ESSAIS:
            row[f"Δt essai {e} (s)"]        = deltas[e]     if e in deltas     else "—"
            row[f"Val. essai {e} (cm H₂O)"] = val_stable[e] if e in val_stable else "—"
        row["Δt moyen (s)"]          = dt_moyen   if dt_moyen   is not None else "—"
        row["Hauteur moy. (cm H₂O)"] = haut_moy   if haut_moy   is not None else "—"
        row["Erreur (%)"]             = erreur_pct if erreur_pct is not None else "—"
        row["Conforme (Oui/Non)"]     = conforme
        rows.append(row)

    return pd.DataFrame(rows)

# 4.  AFFICHAGE CONSOLE
def afficher(tableau: pd.DataFrame) -> None:
    sep  = "=" * 110
    sep2 = "-" * 110

    print(f"\n{sep}")
    print("  RAPPORT - Test T4 : Latence de transmission")
    print(sep)
    print("\n  TABLEAU 3.4.1 - Latence de transmission par palier\n")

    headers = [
        ("Palier",        "(cm H₂O)"),
        ("Δt essai 1",    "(s)"),
        ("Val. essai 1",  "(cm H₂O)"),
        ("Δt essai 2",    "(s)"),
        ("Val. essai 2",  "(cm H₂O)"),
        ("Δt essai 3",    "(s)"),
        ("Val. essai 3",  "(cm H₂O)"),
        ("Δt moyen",      "(s)"),
        ("Hauteur moy.",  "(cm H₂O)"),
        ("Erreur",        "(%)"),
        ("Conforme",      "(Oui/Non)"),
    ]
    col_keys = [
        "Palier (cm H₂O)",
        "Δt essai 1 (s)", "Val. essai 1 (cm H₂O)",
        "Δt essai 2 (s)", "Val. essai 2 (cm H₂O)",
        "Δt essai 3 (s)", "Val. essai 3 (cm H₂O)",
        "Δt moyen (s)", "Hauteur moy. (cm H₂O)",
        "Erreur (%)", "Conforme (Oui/Non)",
    ]
    widths = [9, 11, 13, 11, 13, 11, 13, 11, 14, 9, 10]

    def fmt(v, w):
        s = f"{v:.4f}" if isinstance(v, float) else str(v)
        return s.center(w)

    for line_idx in range(2):
        row_str = "  "
        for (h1, h2), w in zip(headers, widths):
            row_str += (h1 if line_idx == 0 else h2).center(w) + "  "
        print(row_str)

    print("  " + sep2)

    for _, row in tableau.iterrows():
        row_str = "  "
        for key, w in zip(col_keys, widths):
            row_str += fmt(row[key], w) + "  "
        print(row_str)

    print("  " + sep2)

    # Résumé
    all_deltas = []
    for e in ESSAIS:
        col = f"Δt essai {e} (s)"
        if col in tableau.columns:
            all_deltas += [v for v in tableau[col] if isinstance(v, float)]

    lat_moy = round(float(np.mean(all_deltas)),        4) if all_deltas else None
    lat_max = round(float(np.max(all_deltas)),         4) if all_deltas else None
    lat_std = round(float(np.std(all_deltas, ddof=1)), 4) if len(all_deltas) > 1 else 0.0

    n_total     = len(tableau) * len(ESSAIS)
    n_conformes = sum(
        1 for e in ESSAIS
        for v in tableau.get(f"Δt essai {e} (s)", [])
        if isinstance(v, float) and v <= LATENCE_MAX_S
    )

    print(f"\n{sep}")
    print("  Résumé global")
    print(sep)
    print(f"  Latence moyenne    : {lat_moy} s")
    print(f"  Latence maximale   : {lat_max} s")
    print(f"  Écart-type         : {lat_std} s")
    print(f"  Nb essais réussis  : {n_conformes} / {n_total}")
    print(f"  Critère requis     : Δt moyen ≤ {LATENCE_MAX_S} s ({int(LATENCE_MAX_S*1000)} ms)")
    n_oui = (tableau["Conforme (Oui/Non)"] == "Oui").sum()
    n_non = (tableau["Conforme (Oui/Non)"] == "Non").sum()
    print(f"  Paliers conformes  : {n_oui} / {len(tableau)}")
    print(sep)

# POINT D'ENTRÉE
if __name__ == "__main__":
    chemin_csv    = sys.argv[1] if len(sys.argv) > 1 else "pressure_data_T4.csv"
    chemin_export = sys.argv[2] if len(sys.argv) > 2 else None

    df = charger_csv(chemin_csv)
    print(f"[INFO] {len(df)} mesures chargées depuis « {chemin_csv} ».")

    tableau = construire_tableau(df)
    afficher(tableau)
