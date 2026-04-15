"""
=============================================================================
ANALYSE DE VALIDATION - Test T9
Connexion Bluetooth
=============================================================================

Structure du CSV attendu (5 colonnes, séparateur ;) :
  - id           : identifiant de la mesure
  - date         : date/heure de l'acquisition
  - time_s       : temps en secondes depuis le début de l'acquisition
  - mesure_cmH20 : valeur mesurée par le capteur (m)
  - distance     : distance de test en mètres (1, 2, 3, ... 7)

Sorties produites :
  Tableau T9 - pour chaque distance :
    • Nombre de déconnexions  = nb d'intervalles > SEUIL_DECONNEXION_S
    • Messages reçus          = nb de lignes dans le CSV pour cette distance
    • Messages attendus       = round(durée / LOOP_PERIOD_S)
    • Pertes (%)              = (1 − reçus / attendus) × 100

  Résumé :
    • Fréquence réelle (Hz)   = moyenne sur tous les intervalles ≤ 2×LOOP_PERIOD_S
    • Taux de pertes global (%)

Paramètres :
  LOOP_PERIOD_MS      = 50 ms  (20 Hz théorique, défini dans le firmware)
  SEUIL_DECONNEXION_S = 3 × LOOP_PERIOD_S  (intervalle anormalement long)

Utilisation :
  python analyse_test_T9.py                        # lit "pressure_data_T9.csv"
  python analyse_test_T9.py mon_fichier.csv        # lit le fichier spécifié
  python analyse_test_T9.py mon_fichier.csv res.csv  # + export CSV
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd

# PARAMÈTRES DU TEST
LOOP_PERIOD_MS      = 50              # ms  - période firmware (20 Hz)
LOOP_PERIOD_S       = LOOP_PERIOD_MS / 1000.0
FREQ_THEORIQUE      = 1.0 / LOOP_PERIOD_S   # Hz
SEUIL_DECONNEXION_S = 3 * LOOP_PERIOD_S     # s   - Δt > seuil = déconnexion

# 1.  CHARGEMENT ET NETTOYAGE DU CSV
def charger_csv(chemin: str) -> pd.DataFrame:
    if not os.path.exists(chemin):
        sys.exit(f"[ERREUR] Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin, sep=";")
    df.columns = df.columns.str.strip()

    colonnes_requises = {"id", "date", "time_s", "mesure_cmH20", "distance"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        sys.exit(f"[ERREUR] Colonnes manquantes dans le CSV : {manquantes}")

    for col in ["time_s", "mesure_cmH20", "distance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_avant = len(df)
    df = df.dropna(subset=["time_s", "mesure_cmH20", "distance"])
    if len(df) != n_avant:
        print(f"[AVERTISSEMENT] {n_avant - len(df)} ligne(s) ignorée(s).")

    df = df.sort_values(["distance", "time_s"]).reset_index(drop=True)
    return df

# 2.  CALCUL PAR DISTANCE
def analyser_distance(sub: pd.DataFrame) -> dict:
    """
    Pour une distance donnée :

    - Messages reçus    = nb de lignes
    - Durée             = time_s[-1] − time_s[0]
    - Messages attendus = round(durée / LOOP_PERIOD_S)
    - Pertes (%)        = (1 − reçus / attendus) × 100
    - Déconnexions      = nb d'intervalles Δt > SEUIL_DECONNEXION_S
    """
    sub = sub.sort_values("time_s").reset_index(drop=True)
    deltas     = sub["time_s"].diff().dropna()
    duree      = sub["time_s"].iloc[-1] - sub["time_s"].iloc[0]
    n_recus    = len(sub)
    n_attendus = round(duree / LOOP_PERIOD_S)
    pertes_pct = round((1 - n_recus / n_attendus) * 100, 2) if n_attendus > 0 else 0.0
    n_deconnex = int((deltas > SEUIL_DECONNEXION_S).sum())

    return {
        "n_recus":    n_recus,
        "n_attendus": n_attendus,
        "pertes_pct": pertes_pct,
        "n_deconnex": n_deconnex,
        "deltas":     deltas,
    }

# 3.  TABLEAU DES RÉSULTATS
def construire_tableau(df: pd.DataFrame) -> tuple:
    distances = sorted(df["distance"].unique())
    rows = []
    all_deltas = pd.Series(dtype=float)

    for d in distances:
        sub  = df[df["distance"] == d]
        res  = analyser_distance(sub)
        all_deltas = pd.concat([all_deltas,
                                 res["deltas"][res["deltas"] <= 2 * LOOP_PERIOD_S]])
        rows.append({
            "Distance (m)":           int(d) if d == int(d) else d,
            "Nb déconnexions":        res["n_deconnex"],
            "Messages reçus":         res["n_recus"],
            "Messages attendus":      res["n_attendus"],
            "Pertes (%)":             res["pertes_pct"],
        })

    tableau = pd.DataFrame(rows)

    # Statistiques globales
    freq_reelle  = round(1.0 / all_deltas.mean(), 4) if not all_deltas.empty else None
    total_recus  = tableau["Messages reçus"].sum()
    total_attend = tableau["Messages attendus"].sum()
    taux_pertes  = round((1 - total_recus / total_attend) * 100, 2) \
                   if total_attend > 0 else None

    stats = {
        "freq_reelle":  freq_reelle,
        "taux_pertes":  taux_pertes,
        "total_recus":  total_recus,
        "total_attend": total_attend,
    }

    return tableau, stats

# 4.  AFFICHAGE CONSOLE
def afficher(tableau: pd.DataFrame, stats: dict) -> None:
    sep  = "=" * 75
    sep2 = "-" * 75

    print(f"\n{sep}")
    print("  RAPPORT - Test T9 : Connexion Bluetooth")
    print(sep)
    print("\n  TABLEAU 3.9.1 - Résultats par distance\n")

    headers = [
        ("Distance", "(m)"),
        ("Nb", "déconnexions"),
        ("Messages", "reçus"),
        ("Messages", "attendus"),
        ("Pertes", "(%)"),
    ]
    col_keys = [
        "Distance (m)",
        "Nb déconnexions",
        "Messages reçus",
        "Messages attendus",
        "Pertes (%)",
    ]
    widths = [11, 14, 12, 14, 10]

    def fmt(v, w):
        if isinstance(v, float) and v == int(v):
            s = str(int(v))
        elif isinstance(v, float):
            s = f"{v:.2f}"
        else:
            s = str(v)
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

    print(f"\n{sep}")
    print("  Résumé global")
    print(sep)
    print(f"  Fréquence théorique    : {FREQ_THEORIQUE:.1f} Hz  ({LOOP_PERIOD_MS} ms)")
    print(f"  Fréquence réelle       : {stats['freq_reelle']} Hz")
    print(f"  Messages reçus total   : {stats['total_recus']}")
    print(f"  Messages attendus total: {stats['total_attend']}")
    print(f"  Taux de pertes global  : {stats['taux_pertes']} %")
    print(sep + "\n")

# MAIN
if __name__ == "__main__":
    chemin_csv    = sys.argv[1] if len(sys.argv) > 1 else "pressure_data_T9.csv"
    chemin_export = sys.argv[2] if len(sys.argv) > 2 else None

    df = charger_csv(chemin_csv)
    print(f"[INFO] {len(df)} mesures chargées depuis « {chemin_csv} ».")

    tableau, stats = construire_tableau(df)
    afficher(tableau, stats)
