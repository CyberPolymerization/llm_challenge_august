



### Generating a Submission File

Suppse you have come up with an LLM-based method `generate_answer` that generates an answer to a given question. Here is how you can generate a submssion file. Choose a submission id (think team name) for your submission and let it be `AwesomeAnswerGenerator`

```shell
from tqdm import tqdm
from llm_challenge.utils.misc import read_dict_from_json, write_dict_to_json
qas_dict = read_dict_from_json(PATH_TO_QA_JSON_FILE)


answers_dict = {}
for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
    pbar.set_description(f"Answering question {q_id}")
    answer = generate_answer(qa_dict["question"])
    answers_dict[q_id] = answer

sumbission_id = 'AwesomeAnswerGenerator'
write_dict_to_json(f"{submission_id}_train.json", answers_dict)

```
---
### For Submission evaluation

If you'd like to evaluate your submission on your machine on the `train` set:

   ```shell
   python scripts/evaluate.py --submission_json_path $SUBMISSION_FOR_TRAINSET_JSON_PATH --reference_json_fname $PATH_TO_[qas_train.json]
   ```
As you might have guessed, we use the same script to evaluate your sumbissions on the `test` set as follows.

   ```shell
   python scripts/evaluate.py --submission_json_path $SUBMISSION_FOR_TESTSET_JSON_PATH --reference_json_fname $PATH_TO_[qas_test.json]
   ```

You won't be able to run the above command as answers to the test set will not be shared with the participants.
