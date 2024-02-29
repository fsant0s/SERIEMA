from dataclasses import dataclass, field
import json
import logging
from typing import Optional, Tuple, List

import torch
from transformers.training_args import TrainingArguments, requires_backends, cached_property


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )    

@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """    

    data_path: str = field(metadata={
                              'help': 'the path to the csv files containing the dataset. If create_folds is set to True'
                                      'then it is expected that data_path points to one csv containing the entire dataset'
                                      'to split into folds. Otherwise, data_path should be the folder containing'
                                      'train.csv, test.csv, (and val.csv if available)'
                          })
    create_folds: bool = field(default=False,
                               metadata={'help': 'Whether or not we want to create folds for '
                                                 'K fold evaluation of the model'})

    num_folds: int = field(default=5,
                           metadata={'help': 'The number of folds for K fold '
                                             'evaluation of the model. Will not be used if create_folds is False'})
    validation_ratio: float = field(default=0.2,
                                    metadata={'help': 'The ratio of dataset examples to be used for validation across'
                                                      'all folds for K fold evaluation. If num_folds is 5 and '
                                                      'validation_ratio is 0.2. Then a consistent 20% of the examples will'
                                                      'be used for validation for all folds. Then the remaining 80% is used'
                                                      'for K fold split for test and train sets so 0.2*0.8=16%  of '
                                                      'all examples is used for testing and 0.8*0.8=64% of all examples'
                                                      'is used for training for each fold'}
                                    )
    
    freeze_transformer_weights: bool = field(default=False,
                               metadata={
                                   'help': 'To freeze the weights in the encoder layers of a BERT model.'
                               })

    latent_dim: int = field(default=3,
                             metadata={'help': 'Dimension of the latent space'})
    
    bn_enc: int = field(default=False,
                             metadata={'help': 'whether to use batchnorm on VAE encoder'})
    
    bn_dec: int = field(default=False,
                             metadata={'help': 'whether to use batchnorm on VAE decoder'})
   
    vae_out_dim: int = field(default=1,
                             metadata={'help': 'Number of minimum neurons in VAE last layer'})
   
    VAE_architecture: List[int] = field(default_factory=lambda: [1000, 1000, 600, 300],
                           metadata={'help': 'Architecture of a Variational Autoencoder, focusing exclusively on the encoder component.' 
                                     'The corresponding decoder is designed as the inverse of this encoder.'
                                     "Set to 'auto' to use calc_VAE_dims function."})

    column_info_path: str = field(
        default=None,
        metadata={
            'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
    })

    column_info: dict = field(
        default=None,
        metadata={
            'help': 'a dict referencing the text, categorical, numerical, and label columns'
                    'its keys are text_cols, num_cols, and cat_cols'
    })

    categorical_encode_type: str = field(default='ohe',
                                         metadata={
                                             'help': 'sklearn encoder to use for categorical data',
                                             'choices': ['ohe', 'binary', 'label', 'none']
                                         })
    numerical_transformer_method: str = field(default='yeo_johnson',
                                              metadata={
                                                  'help': 'sklearn numerical transformer to preprocess numerical data',
                                                  'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                              })
    
    mlp_division: int = field(default=4,
                              metadata={
                                  'help': 'the ratio of the number of '
                                          'hidden dims in a current layer to the next MLP layer'
                              })
    
    vae_division: int = field(default=2,
                              metadata={
                                  'help': 'the ratio of the number of '
                                          'hidden dims in a current layer to the next VAE layer'
                              })

    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                     metadata={
                                         'help': 'method to combine categorical and numerical features, '
                                                 'see README for all the method'
                                     })
    mlp_dropout: float = field(default=0.1,
                               metadata={
                                 'help': 'dropout ratio used for MLP layers'
                               })
    vae_dropout: float = field(default=0,
                               metadata={
                                 'help': 'dropout ratio used for VAE layers'
                               })
    numerical_bn: bool = field(default=True,
                               metadata={
                                   'help': 'whether to use batchnorm on numerical features'
                               })
    use_simple_classifier: str = field(default=True,
                                       metadata={
                                           'help': 'whether to use single layer or MLP as final classifier'
                                       })
    mlp_act: str = field(default='relu',
                         metadata={
                             'help': 'the activation function to use for finetuning layers',
                             'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                         })
    vae_act: str = field(default='lrelu',
                         metadata={
                             'help': 'the activation function to use for finetuning layers',
                             'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                         })
    gating_beta: float = field(default=0.2,
                               metadata={
                                   'help': "the beta hyperparameters used for gating tabular data "
                                           "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                               })
    
    train_csv_name: str = field(default="train.csv",
                                metadata={'help': 'Train dataset name'}
                                )
    
    val_csv_name: str = field(default="val.csv",
                                metadata={'help': 'Validation dataset name'}
                                )
    
    test_csv_name: str = field(default="test.csv",
                                metadata={'help': 'Test dataset name'}
                                )

    train_samples: int = field(default=0,
                           metadata={'help': 'sample numbers for the training dataset. To use the whole dataset set it to zero'})
    
    val_samples: int = field(default=0,
                           metadata={'help': 'sample numbers for the validation dataset. To use the whole dataset set it to zero'})
    
    test_samples: int = field(default=0,
                           metadata={'help': 'sample numbers for the test dataset. To use the whole dataset set it to zero'})


    def __post_init__(self):
        assert self.column_info != self.column_info_path, 'provide either a path to column_info or a dictionary'
        assert 0 <= self.validation_ratio <= 1, 'validation_ratio must be between 0 and 1'
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)
            assert 'text_cols' in self.column_info
            if 'cat_cols' not in self.column_info:
                self.column_info['cat_cols'] = None
                self.categorical_encode_type = 'none'
            if 'num_cols' not in self.column_info:
                self.column_info['num_cols'] = None
                self.numerical_transformer_method = 'none'
            if 'text_col_sep_token' not in self.column_info:
                self.column_info['text_col_sep_token'] = None

@dataclass
class OurTrainingArguments(TrainingArguments):
    experiment_name: Optional[str] = field(
        default=None,
        metadata={'help': 'A name for the experiment'}
    )

    gpu_num: int = field(
        default=2,
        metadata={'help': 'The gpu number to train on'}
    )

    debug_dataset: bool = field(
        default=False,
        metadata={'help': 'Whether we are training in debug mode (smaller model)'}
    )

    debug_dataset_size: int = field(
        default=100,
        metadata={'help': 'Size of the dataset in debug mode. Only used when debug_dataset = True.'}
    )

    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})

    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Run evaluation during training at each logging step."},
    )

    max_token_length: Optional[int] = field(
        default=None,
        metadata={'help': 'The maximum token length'}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})

    load_best_model_at_end: bool = field(default=True, metadata={"help": "To track the best checkpoint."})

    metric_for_best_model: str = field(default="eval_loss",
        metadata={"help": "Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models."}
    )

    greater_is_better: bool = field(default=False,
        metadata={"help": "Use in conjunction with load_best_model_at_end and metric_for_best_model to specify if better models should have a greater metric or not. "}
    )

    report_to: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "The list of integrations to report the results and logs to."}
    )

    def __post_init__(self):
        if self.debug_dataset:
            self.max_token_length = 16
            self.logging_steps = 5
            self.overwrite_output_dir = True


    @cached_property
    def _setup_devices(self) -> Tuple["torch.device", int]:
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device
    

@dataclass
class ComputerStabilityArguments:
    """
    Arguments pertaining to calculate stability.
    """
    
    clusters: List[int] = field(default_factory=lambda: [2],
                           metadata={'help': 'List of k(s) to stability evaluate.'})

    k_means_random_state: int = field(default=42,
                           metadata={'help': 'Determines random number generation for centroid initialization. Use an int to make the randomness deterministic'})

    n_samples: int = field(default=50,
                           metadata={'help': 'The number of draw bootstrap samples'})
    
    randomStateSeed: int = field(default=1,
                           metadata={'help': 'Random seed used to initialize the pseudo-random number generator or an instantized BitGenerator'})
    
    compute_stability_steps: int = field(default=0,
                           metadata={'help': 'It determines how often (in number of training steps) stability metrics will be recorded in the log during training.'
                                     'Set it to zero to do not compute it'})
    