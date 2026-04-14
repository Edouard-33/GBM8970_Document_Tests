"""
=============================================================================
ANALYSE DE VALIDATION - Test T1
Capteur de pression statique Honeywell ABPDJJT001PDSA3
=============================================================================

Structure du CSV attendu (6 colonnes) :
  - id                : identifiant de la mesure
  - date              : date/heure de l'acquisition (ex: 2026-03-23 14:30:14)
  - time_s            : temps en secondes depuis le début de l'acquisition
  - mesure_cmH20      : valeur brute mesurée par le capteur (cm H₂O)
  - valeur_réel_cmH20 : palier théorique réel appliqué (cm H₂O) - ex: 0, 5, 10...
  - essai             : numéro d'essai (1, 2 ou 3)

Sorties produites :
  1. Tableau T1 - pour chaque palier théorique :
       • Pression moyenne essai 1, 2, 3 (cm H₂O)
       • Pression mesurée moyenne (cm H₂O) = moyenne des moyennes par essai
       • Erreur absolue (cm H₂O) = |pression mesurée moyenne - palier théorique|
       • Erreur relative (%)     = erreur_cm / palier_théorique × 100
                                   (pour palier=0 : erreur_cm / plage_max × 100)
       • Écart-type moyen (moyenne des écarts-types par essai)
       • Conformité : Oui si erreur (%) ≤ ±1.5 %, Non sinon

  2. Fréquence d'échantillonnage :
       • Différences consécutives de time_s  : Δt_i = time_s[i] − time_s[i−1]
       • Temps moyen = mean(Δt_i filtrés ≤ 2 s)  [secondes]
       • Fréquence  = 1 / temps_moyen  [Hz]
       • Conformité : Oui si fréquence ≥ 0.33 Hz

Utilisation :
  python analyse_test_T1.py                          # lit "pressure_data.csv"
  python analyse_test_T1.py mon_fichier.csv          # lit le fichier spécifié
  python analyse_test_T1.py mon_fichier.csv resultats.csv  # + export CSV
=============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd

# PARAMÈTRES DU TEST  (ne pas modifier sans raison)
ERREUR_MAX_CM  = 1.05    # cm H₂O - seuil de conformité sur l'erreur absolue
FREQ_MIN_HZ    = 0.33   # Hz - fréquence d'échantillonnage minimale requise
PLAGE_MAX_CM   = 70.306957829636   # cm H₂O - plage totale du capteur (pour palier = 0)
ESSAIS         = [1, 2, 3]  # numéros d'essais attendus dans le CSV

# 1.  CHARGEMENT ET NETTOYAGE DU CSV
def charger_csv(chemin: str) -> pd.DataFrame:
    """
    Lit le CSV, nettoie les noms de colonnes (espaces), vérifie leur présence
    et retourne un DataFrame prêt à l'emploi.
    """
    if not os.path.exists(chemin):
        sys.exit(f"[ERREUR] Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)

    # Supprimer les espaces éventuels autour des noms de colonnes
    df.columns = df.columns.str.strip()

    colonnes_requises = {"id", "date", "time_s",
                         "mesure_cmH20", "valeur_réel_cmH20", "essai"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        sys.exit(f"[ERREUR] Colonnes manquantes dans le CSV : {manquantes}")

    # Forcer les types numériques
    for col in ["time_s", "mesure_cmH20", "valeur_réel_cmH20", "essai"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_avant = len(df)
    df = df.dropna(subset=["time_s", "mesure_cmH20", "valeur_réel_cmH20", "essai"])
    n_apres = len(df)
    if n_avant != n_apres:
        print(f"[AVERTISSEMENT] {n_avant - n_apres} ligne(s) ignorée(s) (valeurs non numériques).")

    # Trier par time_s pour le calcul de fréquence
    df = df.sort_values("time_s").reset_index(drop=True)

    return df

# 2.  CALCUL DE LA FRÉQUENCE D'ÉCHANTILLONNAGE
def calculer_frequence(df: pd.DataFrame) -> dict:
    """
    Méthode de calcul
    -----------------
    On dispose d'une colonne time_s : temps absolu en secondes de chaque mesure.

    Étape 1 — Calcul des intervalles consécutifs :
        Δt_i = time_s[i] − time_s[i−1]   pour i = 1, 2, …, N−1

    Étape 2 — Temps moyen et fréquence :
        temps_moyen = mean(Δt_i  |  Δt_i ≤ 2 s)   [secondes]
        fréquence   = 1 / temps_moyen               [Hz]

    Retourne un dict avec :
        n_mesures, n_intervalles, temps_moyen_s, frequence_hz, conforme
    """
    deltas = df["time_s"].diff().dropna()                   # N-1 valeurs

    if deltas.empty:
        return {
            "n_mesures":      len(df),
            "n_intervalles":  0,
            "temps_moyen_s":  None,
            "frequence_hz":   None,
            "conforme":       "N/A — aucun intervalle valide",
        }

    temps_moyen = deltas.mean()
    frequence   = 1.0 / temps_moyen if temps_moyen > 0 else np.nan
    conforme    = "Oui" if frequence >= FREQ_MIN_HZ else "Non"

    return {
        "n_mesures":     len(df),
        "n_intervalles": len(deltas),
        "temps_moyen_s": round(float(temps_moyen), 4),
        "frequence_hz":  round(float(frequence),   4),
        "conforme":      conforme,
    }

# 3.  TABLEAU DES RÉSULTATS PAR PALIER
def construire_tableau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque palier théorique distinct présent dans valeur_réel_cmH20 :

    a) Pression moyenne par essai (essai 1, 2, 3) :
           moy_essai_k = mean(mesure_cmH20  où  essai == k  ET  palier == p)

    b) Pression mesurée moyenne :
           pression_moy = mean( moy_essai_1, moy_essai_2, moy_essai_3 )
           (moyenne des moyennes — pondération égale entre essais)

    c) Erreur absolue (cm H₂O) :
           erreur_cm = | pression_moy − palier_théorique |

    d) Erreur relative (%) :
           Si palier > 0 : erreur_pct = erreur_cm / palier_théorique × 100
           Si palier = 0 : erreur_pct = erreur_cm / PLAGE_MAX_CM     × 100
                           (division par la plage totale pour éviter /0)

    e) Écart-type moyen :
           std_essai_k = std(mesure_cmH20  où  essai == k  ET  palier == p)
           ecart_type  = mean( std_essai_k  pour tous les k disponibles )

    f) Conformité :
           "Oui"  si erreur_cm ≤ ERREUR_MAX_CM (1.5 cm H₂O)
           "Non"  sinon
    """
    paliers = sorted(df["valeur_réel_cmH20"].unique())

    rows = []
    for palier in paliers:
        sous = df[df["valeur_réel_cmH20"] == palier]

        # Moyenne et écart-type par essai
        moy_par_essai = {}
        std_par_essai = {}
        for e in ESSAIS:
            vals = sous[sous["essai"] == e]["mesure_cmH20"]
            if not vals.empty:
                moy_par_essai[e] = float(vals.mean())
                std_par_essai[e] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        if not moy_par_essai:
            continue   # pas de données pour ce palier

        # Pression mesurée moyenne (moyenne des moyennes)
        pression_moy = float(np.mean(list(moy_par_essai.values())))

        # Erreur
        erreur_cm = abs(pression_moy - palier)
        if palier != 0:
            erreur_pct = erreur_cm / palier * 100
        else:
            erreur_pct = erreur_cm / PLAGE_MAX_CM * 100

        # Écart-type moyen
        ecart_type_moyen = float(np.mean(list(std_par_essai.values()))) \
                           if std_par_essai else np.nan

        # Conformité
        conforme = "Oui" if erreur_cm <= ERREUR_MAX_CM else "Non"

        # Construction de la ligne du tableau
        row = {"Hauteur (cm H₂O)": int(palier) if palier == int(palier) else palier}
        for e in ESSAIS:
            label = f"Pression moyenne essai {e} (cm H₂O)"
            row[label] = round(moy_par_essai[e], 4) if e in moy_par_essai else "—"
        row["Pression mesurée moyenne (cm H₂O)"] = round(pression_moy,      4)
        row["Erreur (cm H₂O)"]                   = round(erreur_cm,          4)
        row["Erreur (%)"]                         = round(erreur_pct,         4)
        row["Écart-type moyen"]                   = round(ecart_type_moyen,   4)
        row["Conforme (Oui/Non)"]                 = conforme
        rows.append(row)

    return pd.DataFrame(rows)

# 4.  AFFICHAGE CONSOLE
def afficher(tableau: pd.DataFrame, freq: dict) -> None:
    sep  = "=" * 100
    sep2 = "-" * 100

    print(f"\n{sep}")
    print("  Rapport - Test T1 : Précision du capteur Honeywell ABPDJJT001PDSA3")
    print(sep)

    print("\n   TABLEAU — Résultats par palier\n")

    # En-têtes abrégés pour tenir sur la console
    col_map = {
        "Hauteur (cm H₂O)"                    : "Hauteur\n(cm H₂O)",
        "Pression moyenne essai 1 (cm H₂O)"   : "Moy. essai 1\n(cm H₂O)",
        "Pression moyenne essai 2 (cm H₂O)"   : "Moy. essai 2\n(cm H₂O)",
        "Pression moyenne essai 3 (cm H₂O)"   : "Moy. essai 3\n(cm H₂O)",
        "Pression mesurée moyenne (cm H₂O)"   : "Pression moy.\nmesurée (cm H₂O)",
        "Erreur (cm H₂O)"                     : "Erreur\n(cm H₂O)",
        "Erreur (%)"                           : "Erreur\n(%)",
        "Écart-type moyen"                     : "Écart-type\nmoyen",
        "Conforme (Oui/Non)"                   : "Conforme\n(Oui/Non)",
    }

    # Largeurs de chaque colonne (en caractères)
    widths = [9, 13, 13, 13, 18, 10, 9, 12, 10]

    def fmt_val(v, w):
        if isinstance(v, float):
            s = f"{v:.4f}"
        else:
            s = str(v)
        return s.center(w)

    # Ligne d'en-tête (2 lignes par colonne)
    headers = list(col_map.values())
    header_lines = [h.split("\n") for h in headers]
    for line_idx in range(2):
        row_str = "  "
        for i, (parts, w) in enumerate(zip(header_lines, widths)):
            txt = parts[line_idx] if line_idx < len(parts) else ""
            row_str += txt.center(w) + "  "
        print(row_str)

    print("  " + sep2)

    # Lignes de données
    for _, row in tableau.iterrows():
        vals = [
            row["Hauteur (cm H₂O)"],
            row["Pression moyenne essai 1 (cm H₂O)"],
            row["Pression moyenne essai 2 (cm H₂O)"],
            row["Pression moyenne essai 3 (cm H₂O)"],
            row["Pression mesurée moyenne (cm H₂O)"],
            row["Erreur (cm H₂O)"],
            row["Erreur (%)"],
            row["Écart-type moyen"],
            row["Conforme (Oui/Non)"],
        ]
        row_str = "  "
        for v, w in zip(vals, widths):
            row_str += fmt_val(v, w) + "  "
        print(row_str)

    print("  " + sep2)

    print(f"\n{sep}")
    print("  Vérification de la fréquence d'échantillonnage")
    print(sep)
    print(f"  Temps moyen entre deux mesures :  {freq['temps_moyen_s']} s")
    print(f"  Fréquence mesurée :               {freq['frequence_hz']} Hz")
    print(f"  Critère minimal requis :           {FREQ_MIN_HZ} Hz")
    conforme_str = "[X] Oui  [ ] Non" if freq['conforme'] == "Oui" else "[ ] Oui  [X] Non"
    print(f"  Conforme :                         {conforme_str}")
    print(sep)

    # Résumé
    n_oui = (tableau["Conforme (Oui/Non)"] == "Oui").sum()
    n_non = (tableau["Conforme (Oui/Non)"] == "Non").sum()
    print("\n  Résumé global")
    print(f"  Paliers conformes     : {n_oui} / {len(tableau)}")
    print(f"  Paliers non conformes : {n_non} / {len(tableau)}")
    print(f"  Fréquence conforme    : {freq['conforme']}")
    print(sep + "\n")


# MAIN
if __name__ == "__main__":
    # Chemin du CSV en argument ou valeur par défaut
    chemin_csv    = sys.argv[1] if len(sys.argv) > 1 else "pressure_data_T1.csv"
    chemin_export = sys.argv[2] if len(sys.argv) > 2 else None

    # 1. Chargement
    df = charger_csv(chemin_csv)
    print(f"[INFO] {len(df)} mesures chargées depuis « {chemin_csv} ».")

    # 2. Fréquence d'échantillonnage
    freq = calculer_frequence(df)

    # 3. Tableau des résultats
    tableau = construire_tableau(df)

    # 4. Affichage
    afficher(tableau, freq)