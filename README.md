# Hi-DST: A Hierarchical Approach for Scalable and Extensible Dialogue State Tracking

## Install dependencies
python 3.7
```console
❱❱❱ pip install -r requirements.txt
❱❱❱ mkdir data
❱❱❱ mkdir data/mwz2.1
Unzip "data/mwz2.1.7z" file.
```

## Dataset
Download MultiWOZ 2.1 dataset from https://github.com/budzianowski/multiwoz/tree/master/data followed by the required pre-processing. The pre-processed data is already availavle in the "data/mwz2.1" directory. File "data/mwz2.1/data2.2.json" contains the MultiWOZ 2.2 annotations.

## Label variant map
File "trippy_label_variant/multiwoz21.json" contains the label variant map used by TriPpy (https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/blob/master/dataset_config/multiwoz21.json) with few additional variants.

## Prepare data for domain change and domain predictions
```console
❱❱❱ mwz_path=data/mwz2.1
❱❱❱ domain_data_path=data/domain_data
❱❱❱ python create_domain_data.py -path=${mwz_path} -out=${domain_data_path}
```

## Prepare data for slot-action and slot value predictions
```console
❱❱❱ slot_data_path=data/slot_data
❱❱❱ python create_slot_data.py -path=${mwz_path} -out=${slot_data_path}
```

## Train Domain Change Model
```console
❱❱❱ wd_switch=switch_model
❱❱❱ mkdir ${wd_switch}
❱❱❱ python train_switch_model.py -in=${domain_data_path} -path=${wd_switch} -src_file=train_switch_model.py > ${wd_switch}/log.txt
```

## Train Domain Model
```console
❱❱❱ wd_domain=domain_model
❱❱❱ mkdir ${wd_domain}
❱❱❱ python train_domain_model.py -in=${domain_data_path} -path=${wd_domain} -src_file=train_domain_model.py > ${wd_domain}/log.txt
```

## Train Slot Action Model
```console
❱❱❱ wd_slot_act=slot_act_model
❱❱❱ mkdir ${wd_slot_act}
❱❱❱ python train_slot_act.py -in=${slot_data_path} -path=${wd_slot_act} -src_file=train_slot_act.py > ${wd_slot_act}/log.txt
```

## Train Slot Value Model
```console
❱❱❱ wd_slot_val=slot_value_model
❱❱❱ mkdir ${wd_slot_val}
❱❱❱ python train_slot_value.py -in=${slot_data_path} -path=${wd_slot_val} -src_file=train_slot_value.py > ${wd_slot_val}/log.txt
```

## Generate the DST predictions
```console
❱❱❱ model_key=1
❱❱❱ result_dir=result
❱❱❱ python gen_prediction.py -key=${model_key} -out=${result_dir} -switch_path=${wd_switch} -domain_path=${wd_domain} -slot_act_path=${wd_slot_act} -slot_val_path=${wd_slot_val}
```

## Compute Joint Accuracy
```console
❱❱❱ python compute_accuracy.py -key=${model_key} -path=${result_dir}
```

