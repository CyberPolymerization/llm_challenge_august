# helper function to evaluate
from typing import Optional, Dict, Any, List, Union
from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI
from llm_challenge.utils.misc import read_dict_from_json, set_openai_api_key
import numpy as np
from tqdm import tqdm


def load_submission_file(submission_json_fname: str) -> Dict[str, str]:
    return read_dict_from_json(submission_json_fname)


def load_reference_file(
    reference_json_fname: str = "all_qa_dataset.json",
) -> Dict[str, str]:
    return read_dict_from_json(reference_json_fname)


def get_submission_id(submission_json_fname: str) -> str:
    return submission_json_fname.split("/")[-1].split("_")[0]


def build_evaluate_chain():
    # evaluate chain
    llm = ChatOpenAI(temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)
    return eval_chain


def get_submission_grades(
    eval_chain, reference_dict: Dict[str, Dict[str, str]], submission_dict: Dict[str, str]
) -> Dict[str, str]:
    submission_grades = {}
    print("Grading started ...")
    for ref_id, ref_datum in tqdm(reference_dict.items()):
        if ref_id in submission_dict:
            try:
                grade_job_dict = {
                    "query": ref_datum["question"],
                    "answer": ref_datum["answer"],
                    "result": submission_dict[ref_id],
                }
                submission_grades[ref_id] = eval_chain.predict(**grade_job_dict)
            except Exception as e:
                print(f"Grading failed with Exception {e}. Setting grade to UNK")
                submission_grades[ref_id] = "UNK"
        else:
            submission_grades[ref_id] = "INCORRECT"
    print("Grading finished.")
    return submission_grades


def evaluate_submission(
    submission_json_fname: str, reference_json_fname: str = "all_qa_dataset.json"
) -> Dict[str, Union[str, Dict, float]]:
    # set openai api key
    set_openai_api_key()
    # load submission data
    submission_id = get_submission_id(submission_json_fname)
    submission_dict = load_submission_file(submission_json_fname)
    # load reference data
    reference_dict = load_reference_file(reference_json_fname=reference_json_fname)
    # get the grader
    eval_chain = build_evaluate_chain()
    # grade submission again reference
    submission_grades = get_submission_grades(
        eval_chain, reference_dict, submission_dict
    )
    submission_grades_report = {
        "submission_id": submission_id,
        "grades": submission_grades,
        "accuracy": np.mean([_ == "CORRECT" for _ in submission_grades.values()]),
        "ungraded_count": len([_ for _ in submission_grades.values() if _ == "UNK"]),
        "correct_count": len([_ for _ in submission_grades.values() if _ == "CORRECT"]),
        "incorrect_count": len(
            [_ for _ in submission_grades.values() if _ == "INCORRECT"]
        ),
        "total_count": len(submission_grades),
    }
    return submission_grades_report
