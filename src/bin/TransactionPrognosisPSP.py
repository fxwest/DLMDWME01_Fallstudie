"""
Docstring ERKLÄRUNG # TODO
"""

# ----------------------------
# ---------- IMPORT ----------
# ----------------------------
import pickle
import logging as log
from pathlib import Path
from Logger import CustomLogger


# ----------------------------
# -------- PARAMETER ---------
# ----------------------------
FILE_PATH_TRAINED_MODEL = Path(r"../models/gradient_boosting.pkl")


# ----------------------------
# --------- LOAD MODEL -------
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
# ----------- MAIN -----------
# ----------------------------
if __name__ == "__main__":
    CustomLogger()                                                                                                      # Set and load custom logger
    log.info("Starting Credit Card Transaction Tool...")
    loaded_model = load_model(FILE_PATH_TRAINED_MODEL)
    # TODO: Hier GUI und noch Methode um skriptbasiert zu starten
