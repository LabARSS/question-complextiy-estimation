from typing import List


def get_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Write only the answer number and nothing else."
    return sys_msg


option_ids = [str(i + 1) for i in range(20)]


def get_user_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. Write down ONLY the NUMBER of the correct answer and nothing else."
    return user_prompt


estimate_numerical_complexity_system_prompt = 'You are an expert in the topic of the question. Please act as an impartial judge and evaluate the complexity of the multiple-choice question with options below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must not answer the question. You must rate the question complexity as a number from 0 to 1 following the following scale as a reference: high_school_and_easier - 0.0-0.25, undergraduate - 0.25-0.5, graduate - 0.5-0.75, postgraduate - 0.75-1.0. You must return the complexity by strictly following this format: "[[complexity]]", for example: "Complexity: [[0.48]]", which corresponds to a hard question at the undergraduate level.'


valid_nominal_complexities = ["high_school_and_easier", "undergraduate", "graduate", "postgraduate"]
estimate_nominal_complexity_system_prompt = f'You are an expert in the topic of the question. Please act as an impartial judge and evaluate the complexity of the multiple-choice question with options below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must not answer the question. You must rate the question complexity by strictly following the scale: {", ".join(valid_nominal_complexities)}. You must return the complexity by strictly following this format: "[[complexity]]", for example: "Complexity: [[undergraduate]]", which corresponds to the undergraduate level.'
