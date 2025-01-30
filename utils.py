from typing import List


# https://openreview.net/forum?id=shr9PXz7T0
def get_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " You should directly answer the question by choosing the correct option."
    return sys_msg


option_ids = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


# https://openreview.net/forum?id=shr9PXz7T0
def user_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nAnswer:"
    return user_prompt
