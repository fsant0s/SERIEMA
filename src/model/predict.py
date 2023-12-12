
from statistics import mean, stdev

from pprint import pformat

import pandas as pd

from multimodal_transformers.data import load_data_from_folder
import os

from multimodal_exp_args import (
    MultimodalDataTrainingArguments,
    ModelArguments,
    OurTrainingArguments,
    ComputerStabilityArguments,
)

from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import CustomTrainer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments,
)

#from config.definitions import ROOT_DIR, RNDN
ROOT_DIR = "/customer-segmentation-analysis"

for i in [1000, 'all']: #[50, 100, 200, 300, 400, 500, 1000, 'all']

    model_path = ROOT_DIR + f"/datasets/processed/yelp/logs_yelp/val/bertmultilingua_gating_on_cat_and_num_feats_then_sum_model_lr_3e-3_{i}_samples/"
    print(model_path)

    parser = HfArgumentParser(
        (ModelArguments, MultimodalDataTrainingArguments, OurTrainingArguments, ComputerStabilityArguments)
    )

    model_args, data_args, training_args, stability_args = parser.parse_json_file(
            json_file=os.path.abspath(ROOT_DIR + f"/datasets/processed/yelp/logs_yelp/val/bertmultilingua_gating_on_cat_and_num_feats_then_sum_model_lr_3e-3_{i}_samples/train_config.json")
        )

    config = AutoConfig.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    train_dataset, val_dataset, test_dataset = load_data_from_folder(
            data_args.train_csv_name, 
            data_args.val_csv_name, 
            data_args.test_csv_name,
            data_args.train_samples,
            data_args.val_samples,
            data_args.test_samples,
            data_args.data_path,
            data_args.column_info["text_cols"],
            tokenizer,
            categorical_cols=data_args.column_info["cat_cols"],
            numerical_cols=data_args.column_info["num_cols"],
            categorical_encode_type=data_args.categorical_encode_type,
            numerical_transformer_method=data_args.numerical_transformer_method,
            sep_text_token_str=tokenizer.sep_token
            if not data_args.column_info["text_col_sep_token"]
            else data_args.column_info["text_col_sep_token"],
            max_token_length=training_args.max_token_length,
            debug=training_args.debug_dataset,
            debug_dataset_size=training_args.debug_dataset_size,
        )
    train_datasets = [train_dataset]
    val_datasets = [val_dataset]
    test_datasets = [test_dataset]

    tabular_config = TabularConfig(
        cat_feat_dim=train_dataset.cat_feats.shape[1]
        if train_dataset.cat_feats is not None
        else 0,
        numerical_feat_dim=train_dataset.numerical_feats.shape[1]
        if train_dataset.numerical_feats is not None
        else 0,
        **vars(data_args),
    )

    config.tabular_config = tabular_config
    model = AutoModelWithTabular.from_pretrained(
        model_path,
        config = config,
        ignore_mismatched_sizes=True, #Should consider it for cases that categorical features do not appear in test dataset. So, the tabular_combiner (MLP) input/output will be different, because the train may contain features that test does not.
        cache_dir=None, from_tf=False, state_dict=None
    )

    trainer = CustomTrainer(
                model = model,
                args = training_args,
                train_dataset = train_dataset,
                eval_dataset = val_dataset,
                compute_metrics = None)

    predict = trainer.predict(test_dataset).predictions
    pd.DataFrame(predict).to_csv(ROOT_DIR + f"/datasets/processed/yelp/logs_yelp/test/BertBaseUncased-gating_on_cat_and_num_feats_then_sum-{i}_samples_fold_1.csv", index = False)
    break