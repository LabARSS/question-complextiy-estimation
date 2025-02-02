import torch
from typing import Any
from dataclasses import dataclass

LLMModel = Any


# TODO: Cite https://github.com/abazarova/tda4hallucinations/
@dataclass
class TokenwiseEntropy:
    device: str = "cuda"
    llm_model: LLMModel

    def calculate(self, input_ids) -> float:
        token_distribution = self._get_token_distribution(input_ids)
        entropy = self._compute_entropy_from_logits(token_distribution)
        return entropy.detach().cpu().item()

    def _get_token_distribution(self, input_ids) -> torch.Tensor:
        # Yield the output of the model for the current example
        output = self.llm_model(
            input_ids,
            output_hidden_states=True,
            output_attentions=False,
        )

        return output.logits[0, -1:]

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
        return entropies[0]
