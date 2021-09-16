# Adapted from Detectron.pytorch/lib/datasets/dataset_catalog.py
# for this project by Ji Zhang,2019
#-----------------------------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
ANN_FN2 = 'annotation_file2'
ANN_FN3 = 'predicate_file'
ANN_FN4 = 'name_mapping_file'
ANN_FN5 = 'name_list_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    # OpenImages_v4 rel dataset for relationship task
    'oi_rel_train': {
        IM_DIR:
            _DATA_DIR + '/openimages_v4/train',
        ANN_FN:
            _DATA_DIR + '/openimages_v4/rel/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/openimages_v4/rel/rel_only_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json',
    },
    'oi_rel_train_mini': {
        IM_DIR:
            _DATA_DIR + '/openimages_v4/train',
        ANN_FN:
            _DATA_DIR + '/openimages_v4/rel/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/openimages_v4/rel/rel_only_annotations_train_mini.json',
        ANN_FN3:
            _DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json',
    },
    'oi_rel_val': {
        IM_DIR:
            _DATA_DIR + '/openimages_v4/train',
        ANN_FN:
            _DATA_DIR + '/openimages_v4/rel/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/openimages_v4/rel/rel_only_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json',
    },
    'oi_rel_val_mini': {
        IM_DIR:
            _DATA_DIR + '/openimages_v4/train',
        ANN_FN:
            _DATA_DIR + '/openimages_v4/rel/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/openimages_v4/rel/rel_only_annotations_val_mini.json',
        ANN_FN3:
            _DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json',
    },
    # for Kaggle test
    'oi_kaggle_rel_test': {
        IM_DIR:
            _DATA_DIR + '/openimages_v4/rel/kaggle_test_images/challenge2018_test',
        ANN_FN:  # pseudo annotation
            _DATA_DIR + '/openimages_v4/rel/kaggle_test_images/detections_test.json',
        ANN_FN2:
            _DATA_DIR + '/openimages_v4/rel/kaggle_test_images/all_rel_only_annotations_test.json',
        ANN_FN3:
            _DATA_DIR + '/openimages_v4/rel/rel_9_predicates.json',
    },
    # VG dataset
    'vg_train': {
        IM_DIR:
            _DATA_DIR + '/vg/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/vg/rel_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/vg/predicates.json',
    },
    'vg_val': {
        IM_DIR:
            _DATA_DIR + '/vg/VG_100K',
        ANN_FN:
            _DATA_DIR + '/vg/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/vg/rel_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/vg/predicates.json',
    },
    # VRD dataset
    'vrd_train': {
        IM_DIR:
            _DATA_DIR + '/vrd/train_images',
        ANN_FN:
            _DATA_DIR + '/vrd/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/vrd/new_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/vrd/predicates.json',
    },
    'vrd_val': {
        IM_DIR:
            _DATA_DIR + '/vrd/val_images',
        ANN_FN:
            _DATA_DIR + '/vrd/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/vrd/new_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/vrd/predicates.json',
    },
    # vidvrd
    'vidvrd_train': {
        IM_DIR:
            _DATA_DIR + '/vidvrd/frames',
        ANN_FN:
            _DATA_DIR + '/vidvrd/annotations/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/vidvrd/annotations/new_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/vidvrd/annotations/predicates.json',
        ANN_FN4:
            _DATA_DIR + '/vidvrd/annotations/train_fname_mapping.json',
        ANN_FN5:
            _DATA_DIR + '/vidvrd/annotations/train_fname_list.json',
    },
    'vidvrd_val': {
        IM_DIR:
            _DATA_DIR + '/vidvrd/frames',
        ANN_FN:
            _DATA_DIR + '/vidvrd/annotations/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/vidvrd/annotations/new_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/vidvrd/annotations/predicates.json',
        ANN_FN4:
            _DATA_DIR + '/vidvrd/annotations/val_fname_mapping.json',
        ANN_FN5:
            _DATA_DIR + '/vidvrd/annotations/val_fname_list.json',
    },
    # ActionGenome dataset
    'ag_train': {
        IM_DIR:
            _DATA_DIR + '/ag/frames',
        ANN_FN:
            _DATA_DIR + '/ag/annotations/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/ag/annotations/new_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/ag/annotations/predicates.json',
        ANN_FN4:
            _DATA_DIR + '/ag/annotations/train_fname_mapping.json',
        ANN_FN5:
            _DATA_DIR + '/ag/annotations/train_fname_list.json',
    },
    'ag_val': {
        IM_DIR:
            _DATA_DIR + '/ag/frames',
        ANN_FN:
            _DATA_DIR + '/ag/annotations/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/ag/annotations/new_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/ag/annotations/predicates.json',
        ANN_FN4:
            _DATA_DIR + '/ag/annotations/val_fname_mapping.json',
        ANN_FN5:
            _DATA_DIR + '/ag/annotations/val_fname_list.json',
    },
    # GQA dataset
    'gqa_train': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_train.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_train.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    'gqa_val': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_val.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_val.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    'gqa_all': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    'gqa_1st_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_1st_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    'gqa_2nd_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_2nd_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    'gqa_3rd_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_3rd_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships.json',
    },
    # GQA no_plural_verb dataset
    'gqa_verb_train': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_train_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_verb_no_plural_train.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_val': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_val_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_verb_no_plural_val.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_all': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_1st_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_1st_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_2nd_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_2nd_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_3rd_of_3': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_3rd_of_3.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_1st_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_1st_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_2nd_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_2nd_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_3rd_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_3rd_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_4th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_4th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_5th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_5th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    'gqa_verb_6th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_6th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_verb.json',
    },
    # GQA no_plural_spt dataset
    'gqa_spt_train': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_train_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_spt_no_plural_train.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_spt.json',
    },
    'gqa_spt_val': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_val_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_spt_no_plural_val.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_spt.json',
    },
    # GQA no_plural_misc dataset
    'gqa_misc_train': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_train_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_misc_no_plural_train.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_val': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/detections_val_no_plural.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/rel_annotations_misc_no_plural_val.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_1st_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_1st_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_2nd_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_2nd_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_3rd_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_3rd_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_4th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_4th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_5th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_5th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
    'gqa_misc_6th_of_6': {
        IM_DIR:
            _DATA_DIR + '/gqa/images',
        ANN_FN:
            _DATA_DIR + '/gqa/dummy_detections_no_plural_all.json',
        ANN_FN2:
            _DATA_DIR + '/gqa/dummy_rel_annotations_all_6th_of_6.json',
        ANN_FN3:
            _DATA_DIR + '/gqa/relationships_misc.json',
    },
}
