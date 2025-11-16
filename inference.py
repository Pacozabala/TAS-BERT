import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from processor import Semeval_Processor
import tokenization
from TAS_BERT_joint import convert_examples_to_features
from modeling import BertConfig, BertForTABSAJoint, BertForTABSAJoint_CRF


def to_tensor_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ner_label_ids = torch.tensor([f.ner_label_ids for f in features], dtype=torch.long)
    all_ner_mask = torch.tensor([f.ner_mask for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ner_label_ids, all_ner_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--bert_config_file", required=True)
    parser.add_argument("--init_checkpoint", required=True, help="pytorch checkpoint path or bert weights")
    parser.add_argument("--output_file", default="predictions.txt")
    parser.add_argument("--tokenize_method", choices=["prefix_match","unk_replace","word_split"], default="word_split")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Semeval_Processor()
    label_list = processor.get_labels()
    ner_label_list = processor.get_ner_labels(args.data_dir)

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, tokenize_method=args.tokenize_method, do_lower_case=args.do_lower_case)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    # build model
    if args.use_crf:
        model = BertForTABSAJoint_CRF(bert_config, len(label_list), len(ner_label_list))
    else:
        model = BertForTABSAJoint(bert_config, len(label_list), len(ner_label_list), args.max_seq_length)

    # Try loading either full model checkpoint or BERT weights into model.bert
    ckpt = torch.load(args.init_checkpoint, map_location='cpu')
    try:
        # try to load full model state dict
        model.load_state_dict(ckpt)
    except Exception:
        try:
            model.bert.load_state_dict(ckpt)
        except Exception as e:
            # fallback: if checkpoint is a nested dict like {'bert':..., 'classifier':...}
            if isinstance(ckpt, dict):
                loaded = False
                if 'bert' in ckpt and isinstance(ckpt['bert'], dict):
                    try:
                        model.bert.load_state_dict(ckpt['bert'])
                        loaded = True
                    except Exception:
                        loaded = False
                if not loaded:
                    raise RuntimeError("Failed to load checkpoint: try converting TF checkpoint or provide a matching PyTorch checkpoint")

    model.to(device)
    model.eval()

    # test dataset
    test_examples = processor.get_test_examples(args.data_dir)
    test_features, test_tokens = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, ner_label_list, args.tokenize_method)
    test_dataset = to_tensor_dataset(test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        # write human-readable labels instead of numeric IDs
        fout.write('true_label\tpred_label\tsentence\ttrue_ner\tpredict_ner\n')
        batch_index = 0
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, ner_label_ids, ner_mask = batch

            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, label_ids, ner_label_ids, ner_mask)
                # model forward returns (loss, ner_loss, logits, ner_logits) for joint softmax/CRF variants
                if args.use_crf:
                    # BertForTABSAJoint_CRF returns (loss, ner_loss, logits, ner_predict)
                    _, _, logits, ner_predict = outputs
                    ner_logits = ner_predict
                else:
                    _, _, logits, ner_logits_tensor = outputs
                    ner_logits = torch.argmax(F.log_softmax(ner_logits_tensor, dim=2), dim=2).detach().cpu().numpy()

            logits = F.softmax(logits, dim=-1).detach().cpu().numpy()
            label_ids_np = label_ids.cpu().numpy()
            outputs_np = np.argmax(logits, axis=1)

            ner_label_ids_np = ner_label_ids.cpu().numpy()

            # ner_predict for CRF may already be a list/array shaped [batch, seq_len]
            if args.use_crf:
                ner_preds = ner_logits
            else:
                ner_preds = ner_logits

            # test_tokens holds tokens for whole set: slice the batch
            batch_tokens = test_tokens[batch_index*args.eval_batch_size:(batch_index+1)*args.eval_batch_size]
            batch_index += 1

            for i in range(len(batch_tokens)):
                sentence_tokens = batch_tokens[i]
                sent_true = []
                sent_pred = []
                clean_tokens = []
                for j, tok in enumerate(sentence_tokens):
                    if not tok.startswith('##'):
                        clean_tokens.append(tok)
                        true_tag = ner_label_list[ner_label_ids_np[i][j]]
                        # handle different types for preds
                        pred_id = int(ner_preds[i][j]) if not isinstance(ner_preds[i][j], str) else ner_label_list.index(ner_preds[i][j])
                        pred_tag = ner_label_list[pred_id]
                        sent_true.append(true_tag)
                        sent_pred.append(pred_tag)

                true_label_name = label_list[label_ids_np[i]]
                pred_label_name = label_list[outputs_np[i]]

                fout.write(f"{true_label_name}\t{pred_label_name}\t{' '.join(clean_tokens)}\t{' '.join(sent_true)}\t{' '.join(sent_pred)}\n")

    print("Predictions written to", args.output_file)


if __name__ == "__main__":
    main()
