from typing import List, Dict

import numpy as np
import mindspore as ms

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = ms.Tensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = ms.Tensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], ms.Tensor) else first['label']
        dtype = ms.int64 if isinstance(label, int) else ms.float_
        batch['labels'] = ms.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], ms.Tensor):
            batch['labels'] = ms.ops.stack([f['label_ids'] for f in features])
        else:
            dtype = ms.int64 if isinstance(first['label_ids'][0], int) else ms.float_
            batch['labels'] = ms.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and not isinstance(v, str):
            if isinstance(v, ms.Tensor):
                batch[k] = ms.ops.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = ms.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = ms.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features: List[Dict[str, ms.Tensor]], batch_info):
    # pad_id = 0

    batch_size = len(features)

    batch_lens = [feat['input_ids'].shape for feat in features]
    # all element of batch_lens is the same in static shape
    max_item_length = max(batch_lens)[0]
    for idx in range(batch_size):
        feat = features[idx]
        feat['img_context_token_index'] += max_item_length * idx
        # feat['labels'] = feat['labels'].astype(np.int32)

        # No need to pad in static shape
        # temp_input_ids = np.array([pad_id] * max_item_length)
        # temp_input_ids[:input_ids[idx].shape[0]] = input_ids[idx]
        # input_ids[idx] = temp_input_ids
        # temp_labels = np.array([IGNORE_INDEX] * max_item_length)
        # temp_labels[:labels[idx].shape[0]] = labels[idx]
        # labels[idx] = temp_labels.astype(np.int32)
        # attention_mask[idx] = input_ids[idx] != pad_id

    input_ids =  np.stack([feat['input_ids'] for feat in features])
    labels = np.stack([feat['labels'] for feat in features])
    attention_mask = np.stack([feat['attention_mask'] for feat in features])
    pixel_values = np.concatenate([feat['pixel_values'] for feat in features])
    image_flags = np.concatenate([feat['image_flags'] for feat in features])
    img_context_token_index = np.concatenate([feat['img_context_token_index'] for feat in features])

    return input_ids, labels, attention_mask, pixel_values, image_flags, img_context_token_index