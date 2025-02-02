from typing import Literal, Any
import typing as tp

import pandas as pd
import torch
from tqdm import trange

from dataclasses import dataclass

T = list[float]
LLMModel = Any
Tokenizer = Any


# TODO: Cite https://github.com/abazarova/tda4hallucinations/
@dataclass
class TokenwiseEntropy:
    """
    A class to compute token-wise entropy using pre-trained models.

    This class provides methods to calculate token-wise entropy for given text prompts and responses using
    pre-trained language models. It supports different aggregation methods for combining entropy values and
    allows saving and loading of precomputed entropy results.
    """

    aggregation: tp.Literal["max", "min", "mean"]
    dtype: Literal["float32", "float16", "bfloat16"] = ("float16",)
    device: str = "cuda"
    llm_model: LLMModel
    tokenizer: Tokenizer

    def calculate(self, X: pd.DataFrame) -> list[float]:
        """
        Calculate token-wise entropy for each entry in the DataFrame.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing 'prompt' and 'response' columns.

        Returns:
        -------
        list
            List of entropy values for each entry.
        """
        token_distributions = self._get_token_distributions(X)

        entropies_list = []

        for token_distribution in token_distributions:
            entropies_list.append(self._compute_entropy_from_logits(token_distribution))

        aggregated_entropies = [self._aggregate_entropies(entropies) for entropies in entropies_list]
        return aggregated_entropies

    def _aggregate_entropies(self, entropies: torch.Tensor) -> float:
        # Aggregate the entropies
        if self.aggregation == "max":
            aggregated_entropy = max(entropies)
        elif self.aggregation == "min":
            aggregated_entropy = min(entropies)
        elif self.aggregation == "mean":
            aggregated_entropy = sum(entropies) / len(entropies)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_entropy

    def _get_token_distributions(
        self,
        X: pd.DataFrame,
    ) -> list[torch.Tensor]:
        """Retrieve the distribution on each token of the model for each input in the DataFrame."""
        logits_list = []

        for i in trange(len(X)):
            prompt = X["prompt"].iloc[i]
            answer = X["response"].iloc[i]

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            answer_ids = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
            input_ids = torch.cat([prompt_ids["input_ids"], answer_ids["input_ids"]], axis=1).to(self.device)

            # Yield the output of the model for the current example
            output = self.llm_model(
                input_ids,
                output_hidden_states=True,
                output_attentions=False,
            )

            len_answer = len(answer_ids[0])
            logits_list.append(output.logits[0, -len_answer:].cpu())

        return logits_list

    def _compute_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from logits.

        Parameters:
        ----------
        logits : torch.Tensor
            Logits from the model.

        Returns:
        -------
        torch.Tensor
            Entropy values.
        """
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log(probabilities + 1e-12)
        entropies = -torch.sum(probabilities * log_probabilities, dim=-1)
        return entropies
