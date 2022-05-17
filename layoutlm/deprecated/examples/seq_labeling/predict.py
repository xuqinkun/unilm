# -*- coding: utf-8 -*-
import os
from eval import evaluate
from transformers import logger


def do_predict(args, tokenizer_class, model_class, labels, pad_token_label_id):
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(args.output_dir)
    model.to(args.device)
    results, predictions = evaluate(
        args, model, tokenizer, labels, pad_token_label_id, mode="test"
    )
    # Save results
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(
        args.output_dir, "test_predictions.txt"
    )
    with open(output_test_predictions_file, "w", encoding="utf8") as writer:
        with open(
                os.path.join(args.data_dir, "test.txt"), "r", encoding="utf8"
        ) as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = (
                            line.split()[0]
                            + " "
                            + predictions[example_id].pop(0)
                            + "\n"
                    )
                    writer.write(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0],
                    )
    return results