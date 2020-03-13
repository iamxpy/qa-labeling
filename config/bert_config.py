import os
cfg = {}

cfg["NFOLDS"] = 5
cfg["EPOCH"] = 5
cfg["EVAL_EVERY"] = 200
cfg["MAX_SEQUENCE_LENGTH"] = 512
cfg["swa_alpha"] = 0.99
cfg["Last_Layer"] = 4
cfg["Save_ckpt"] = True
cfg["Use_semi_supervised"] = False
cfg["LR"] = 3e-5
cfg["choose"] = "both"
cfg["ratio"] = 2.0 / 3
cfg["BATCH_SIZE"] = 2
cfg["output"] = "exp59"
# bert_large_cased_whole_word_masking,bert-base-uncased, albert-xxlarge-v2,bert_large_uncased_whole_word_masking,roberta-base,roberta-large
cfg["pretrained_model"] = "roberta-base"
cfg["model_dir"] = "./torch-bert-weights/" + cfg["pretrained_model"]
cfg["weight"] = None
cfg["question"] = True

cfg["upsample"] = 1

if "albert" in cfg["pretrained_model"]:
    cfg["NUM_LAYERS"] = 12
    cfg["hidden_size"] = 4096
elif "large" in cfg["pretrained_model"]:
    cfg["NUM_LAYERS"] = 24
    cfg["hidden_size"] = 1024
else:
    cfg["hidden_size"] = 768
    cfg["NUM_LAYERS"] = 12
