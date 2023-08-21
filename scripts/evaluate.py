"""
A script to evaluate a submission json file given reference json file.
"""
import sys
sys.path.append("/Users/arnelcatangay/Documents/GitHub/llm_challenge_main")

import argparse
from pathlib import Path
from llm_challenge.evaluation import evaluate_submission
from llm_challenge.utils.misc import write_dict_to_json


def main(args):

    if args.submission_json_path.is_dir():
        fnames = args.submission_json_path.glob("/*.json")
    else:
        fnames = [args.submission_json_path]
    
    all_results = []
    for fn in fnames:
        submission_result = evaluate_submission(
            submission_json_fname=str(fn),
            reference_json_fname=args.reference_json_fname,
        )
        all_results.append(submission_result)
    print(all_results)
    if args.submission_evaluation_json_fname is not None:
        write_dict_to_json(args.submission_evaluation_json_fname, all_results)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""" 
    Use this script to evaluate your submission.
    """
    )
    parser.add_argument(
        "--submission_json_path",
        type=Path,
        help="path to the submission json file or folder containing submission json files",
        required=True,
    )
    parser.add_argument(
        "--reference_json_fname", type=str, default="../notebooks/simple_qa_pairs.json",
        help="path to the reference json file, participants need not change this."
    )
    parser.add_argument(
        "--submission_evaluation_json_fname", type=str, default="../submissions_evaluation.json",
        help="path to the evaluation report json file to be created by the script."
    )
    args = parser.parse_args()
    main(args)
