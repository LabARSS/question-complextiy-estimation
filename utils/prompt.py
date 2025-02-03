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
