import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
class APPOCritic(nn.Module):
    """
    APPOCritic is a neural network module designed to estimate the value function
    for the APPO (Actor-Proximal Policy Optimization) reinforcement learning algorithm.
    It leverages a pre-trained transformer-based language model (the actor) to extract
    contextual embeddings, which are then processed through a value head to produce
    scalar value estimates.

    Attributes:
        config (object): Configuration of the base transformer model.
        num_padding_at_beginning (int): Number of padding tokens at the beginning of input sequences.
        v_head (nn.Linear): Linear layer mapping from embedding dimension to scalar value (used for models with 'word_embed_proj_dim').
        v_head_mlp1 (nn.Linear): First linear layer of the MLP value head (used for models without 'word_embed_proj_dim').
        v_head_mlp2 (nn.Linear): Second linear layer of the MLP value head.
        v_head_mlp3 (nn.Linear): Final linear layer of the MLP value head producing scalar value estimates.
        relu (nn.ReLU): ReLU activation function used in the MLP.
        rwtranrsformer (nn.Module): The base transformer model (actor) used for feature extraction.
        PAD_ID (int): Token ID used for padding sequences.
    """
    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        """
        Initializes the APPOCritic module.

        Args:
            base_model (nn.Module): The pre-trained transformer-based language model (actor) used for feature extraction.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the base model.
            num_padding_at_beginning (int, optional): Number of padding tokens at the beginning of input sequences. Defaults to 0.

        Raises:
            NotImplementedError: If the base model's configuration does not have 'word_embed_proj_dim' and lacks 'hidden_size' or 'n_embd'.
        """
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
            self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
            self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
            self.relu = nn.ReLU()
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        """
        Enables gradient checkpointing on the base transformer model.
        Gradient checkpointing reduces memory usage by trading compute for memory,
        which can be beneficial during training.
        """
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """
        Disables gradient checkpointing on the base transformer model.
        This restores normal memory usage and computational behavior.
        """
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=False
        ):
        """
        Performs a forward pass through the APPOCritic to estimate the value function.

        Args:
            input_ids (torch.LongTensor, optional): Token IDs for input sequences. Shape: (batch_size, sequence_length).
            attention_mask (torch.FloatTensor, optional): Attention mask indicating non-padded tokens. Shape: (batch_size, sequence_length).
            past_key_values (tuple, optional): Cached key and value states for faster generation. Defaults to None.
            head_mask (torch.FloatTensor, optional): Masking tensor for attention heads. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed token embeddings. Defaults to None.
            use_cache (bool, optional): Whether to use cached key and value states. Defaults to False.

        Returns:
            torch.FloatTensor: Estimated value function for each input in the batch. Shape: (batch_size,).
        """
        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True
                )

        # Extract the last hidden state of the final token from the last transformer layer
        hidden_states = transformer_outputs.hidden_states[-1][:, -1, :].float()

        # Pass the hidden states through the value head to obtain scalar value estimates
        if hasattr(self, "v_head"):
            # If using a single linear layer for value estimation
            values = self.v_head(hidden_states).squeeze(-1)
        else:
            # If using an MLP for value estimation
            x = self.relu(self.v_head_mlp1(hidden_states))
            x = self.relu(self.v_head_mlp2(x))
            values = self.v_head_mlp3(x).squeeze(-1)

        return values
    

class TPPOCritic(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
            self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
            self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
            self.relu = nn.ReLU()
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      use_cache=False):
        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True)

        hidden_states = transformer_outputs[1][-1].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x)
        return values


class ETPOCritic(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
            self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
            self.v_head_mlp3 = nn.Linear(512, self.config.vocab_size, bias=False)
            self.relu = nn.ReLU()
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      head_mask=None,
                      inputs_embeds=None,
                      use_cache=False):
        with torch.no_grad():
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True)
            
        hidden_states = transformer_outputs[1][-1].float()

        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x)
        return values
