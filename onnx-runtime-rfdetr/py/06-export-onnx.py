from rfdetr import (
    RFDETRBase,
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge
)


model_dict = {
    "base": RFDETRBase,
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}

# -- Configurations
MODEL_SIZE = "base"
CHECKPOINT_PATH = "./output/checkpoint_best_total.pth"
OUT_DIR = "assets/models"


def main():
    model = model_dict.get(MODEL_SIZE)(pretrain_weights=CHECKPOINT_PATH)
    model.optimize_for_inference()
    model.export(output_dir=OUT_DIR)
    return


if __name__ == "__main__":
    main()
