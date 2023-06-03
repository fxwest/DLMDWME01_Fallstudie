"""
Docstring ERKLÄRUNG # TODO
"""

# ----------------------------
# ---------- IMPORT ----------
# ----------------------------
import pickle
import numpy as np
import pandas as pd
import logging as log
from pathlib import Path
from Logger import CustomLogger


# ----------------------------
# -------- PARAMETER ---------
# ----------------------------
FILE_PATH_TRAINED_MODEL = Path(r"../models/gradient_boosting.pkl")
MAX_PROBA_DELTA = 0.10                                                                                                  # Max probability delta to allow selection of cheaper PSP
MIN_PROBA_DELTA = 0.15                                                                                                  # Min probability delta to allow selection of more expensive PSP
AVAILABLE_PSP = [
    {
        "psp_name": "Moneycard",
        "fee_successfully": 5.0,
        "fee_failed": 2.0,
    },
    {
        "psp_name": "Goldcard",
        "fee_successfully": 10.0,
        "fee_failed": 5.0,
    },
    {
        "psp_name": "UK_Card",
        "fee_successfully": 3.0,
        "fee_failed": 1.0,
    },
    {
        "psp_name": "Simplecard",
        "fee_successfully": 1.0,
        "fee_failed": 0.5,
    }
]


# ----------------------------
# ------ SUB FUNCTIONS -------
# ----------------------------
def load_model(model_path):
    """
    This method loads the trained machine learning model.
    :param model_path: Path to the model.pkl
    :return: Returns the loaded model.
    """
    log.info(f"Loading trained model from: \"{model_path.as_posix()}\"")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# ----------------------------
# ---------- MAIN ------------
# ----------------------------
def select_psp(transaction_hour, transaction_amount, is_secured, is_weekend, model_path=FILE_PATH_TRAINED_MODEL):
    CustomLogger()                                                                                                      # Set and load custom logger
    log.info("Starting Credit Card Transaction Tool...")
    loaded_model = load_model(model_path)

    # --- Estimate PSP success probabilities
    success_probability = []
    y_pred_proba_list = []
    for psp in AVAILABLE_PSP:
        log.info(f"Estimating probability for successful {psp['psp_name']} transaction...")
        goldcard_flag = psp["psp_name"] == "Goldcard"
        simplecard_flag = psp["psp_name"] == "Simplecard"
        ukcard_flag = psp["psp_name"] == "UK_Card"

        input_data = [[transaction_hour, transaction_amount, is_secured, is_weekend, goldcard_flag, simplecard_flag, ukcard_flag]]
        input_df = pd.DataFrame(input_data, columns=['hour', 'amount', '3D_secured', 'is_weekend', 'Goldcard', 'Simplecard', 'UK_Card'])
        y_pred = loaded_model.predict(input_df)
        y_pred_proba = loaded_model.predict_proba(input_df)[:, 1]
        if y_pred:
            log.info(f"Successful {psp['psp_name']} transaction with {round(y_pred_proba[0] * 100, 1)}% success probability.")
            success_probability.append(y_pred_proba[0])
        else:
            log.info(f"Failing {psp['psp_name']} transaction with {round(y_pred_proba[0] * 100, 1)}% success probability.")
            success_probability.append(False)
        y_pred_proba_list.append(y_pred_proba[0])

    # --- Select best PSP
    num_success_pred = sum(1 for proba in success_probability if proba)
    if num_success_pred == 0:
        # --- Select cheapest PSP, because no PSP has a positive prediction for success
        log.warning(f"None of the available PSP has a positive prediction for success. Selecting the PSP with the cheapest failed fee...")
        min_failed_fee_psp = min(AVAILABLE_PSP, key=lambda x: x["fee_failed"])
        selected_psp = min_failed_fee_psp
        idx = next((index for (index, d) in enumerate(AVAILABLE_PSP) if d["psp_name"] == selected_psp["psp_name"]), None)
        selected_proba = y_pred_proba_list[idx]

        # --- Select more expensive PSP with higher probability if delta is big enough
        max_proba = max(y_pred_proba_list)
        max_idx = np.array(y_pred_proba_list).argmax()
        delta_proba = max_proba - selected_proba
        if delta_proba > MIN_PROBA_DELTA:
            log.info(f"Selecting a more expensive PSP because the probability for success is {round(delta_proba * 100, 1)}% higher.")
            selected_psp = AVAILABLE_PSP[max_idx]
            selected_proba = max_proba
    elif num_success_pred > 0:
        # --- Select the PSP with the highest probability for success
        max_proba = max(success_probability)
        max_idx = np.array(success_probability).argmax()
        psp_name = AVAILABLE_PSP[max_idx]["psp_name"]
        fee_successfully = AVAILABLE_PSP[max_idx]["fee_successfully"]
        fee_failed = AVAILABLE_PSP[max_idx]["fee_failed"]
        selected_psp = AVAILABLE_PSP[max_idx]
        selected_proba = max_proba
        if num_success_pred > 1:
            # --- Select the PSP with the second-highest probability for success if the delta is not too big and if the other PSP is cheaper
            temp_success_probability = success_probability.copy()
            temp_success_probability.remove(max_proba)
            sec_max_proba = max(temp_success_probability)
            sec_max_idx = success_probability.index(sec_max_proba)
            sec_psp_name = AVAILABLE_PSP[sec_max_idx]["psp_name"]
            sec_fee_successfully = AVAILABLE_PSP[sec_max_idx]["fee_successfully"]
            sec_fee_failed = AVAILABLE_PSP[sec_max_idx]["fee_failed"]
            log.info(f"Highest success probability has {psp_name} {round(max_proba * 100, 1)}% and a success fee of {fee_successfully}€ and a fail fee of {fee_failed}€.")
            log.info(f"Second-highest success probability has {sec_psp_name} {round(sec_max_proba * 100, 1)}% and a success fee of {sec_fee_successfully}€ and a fail fee of {sec_fee_failed}€.")
            delta_proba = max_proba - sec_max_proba
            if delta_proba <= MAX_PROBA_DELTA and sec_fee_failed < fee_successfully and sec_fee_successfully < fee_successfully:
                selected_psp = AVAILABLE_PSP[sec_max_idx]
                selected_proba = sec_max_proba
                log.info(f"Selecting PSP with second-highest success probability, because its cheaper and the probability for success is only {round(delta_proba * 100, 1)}% lower.")

    return selected_psp, selected_proba


if __name__ == "__main__":
   selected_psp, selected_proba = select_psp(transaction_hour=14,
                                             transaction_amount=80.9,
                                             is_secured=True,
                                             is_weekend=True)
   log.info(f"Selected {selected_psp['psp_name']} with a success probability of {round(selected_proba * 100, 1)}% and a success fee of {selected_psp['fee_successfully']}€ and a fail fee of {selected_psp['fee_failed']}€.")
   # TODO: Feedback if transaction was successful and save to database for further training and model improvement
   # TODO: Track fees of successful and failed transactions for visualization in grafana
   